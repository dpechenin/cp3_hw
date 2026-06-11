"""
Вариант 03 (рыба). Печенин Данила БПМ233.

Существо: рыба, вид сверху. Тело построено как сглаженное объединение
окружностей разных радиусов, нанизанных на узлы одного позвоночника. У рыбы есть
пара грудных плавников (пара конечностей), хвостовой плавник и пара глаз.

1. Элементы анимации
1.1. Фон (вода)
Вертикальный градиент: сверху более светлый синий, снизу темно-синий.

1.2. Позвоночник
Цепочка из N узлов фиксированной длины сегмента (нерастяжимый позвоночник).
Голова (узел 0) инерционно догоняет курсор мыши, остальные узлы подтягиваются
к предыдущему по правилу follow the leader. Поверх решения накладывается
бегущая волна, благодаря которой рыба всегда живо извивается даже
при неподвижном курсоре.

1.3. Тело
Объединение усеченных конусов (с разным радиусом) вдоль сегментов позвоночника
через сглаживающий минимум. Радиус по длине тела задан так:
толще у головы, тоньше к хвосту.

1.4. Грудные плавники (пара конечностей)
Два эллиптических плавника, прикрепленных к узлу в передней трети тела. Угол
взмаха складывается из колебания во времени и реакции на кривизну позвоночника.

1.5. Хвостовой плавник
Два расходящихся эллиптических лепестка у последнего узла, слегка колеблются.

1.6. Глаза (пара)
Два белых глаза с темными зрачками и бликом возле головы, смещены перпендикулярно
направлению головы.

1.7. Цвет слоями
Каждый элемент (хвост, плавники, тело, глаза) - отдельный цветной слой,
итоговое изображение собирается последовательным смешиванием слоев.

2. Взаимосвязи элементов
Сначала kernel compute двигает голову к курсору (инерция), затем последовательно
выравнивает длины сегментов (нерастяжимость), затем ограничивает минимальный
угол между сегментами, и в конце считает отображаемые позиции узлов с бегущей
волной. Kernel render для каждого пикселя строит цвет: фон, потом хвостовой
плавник, грудные плавники, тело и глаза - каждый своим слоем поверх предыдущего.
Положение и ориентация плавников и глаз берутся из отображаемых узлов, поэтому
они жестко привязаны к позвоночнику.

3. Идея реализации
- позвоночник считается на CPU/GPU один раз за кадр в сериализованном цикле,
  это и есть шаг расчета, отрисовка - отдельный шаг render
- тело удобно задать как сглаженное объединение усеченных конусов, это
  автоматически дает сглаженные окружности разных радиусов
- инерция головы (затухающая пружина) дает запаздывание из модификации 1б
- ограничение угла между сегментами не дает телу складываться на резких
  разворотах (модификация 2)
- плавники реагируют и на время, и на кривизну позвоночника (модификация 4)
- раздельные цветные слои и их смешивание дают цветное существо (модификация 5)
"""

from typing import Optional, Tuple
import math
import time

import taichi as ti
import taichi.math as tm


@ti.func
def smooth(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    """Гладкая интерполяция Эрмита (аналог smoothstep) между двумя границами."""
    h = tm.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return h * h * (3.0 - 2.0 * h)


@ti.func
def alpha_from_sdf(distance: ti.f32, aa: ti.f32) -> ti.f32:
    """Перевести знаковое расстояние в сглаженное покрытие пикселя [0, 1]."""
    return 1.0 - smooth(-aa, aa, distance)


@ti.func
def blend(bottom: tm.vec3, top: tm.vec3, alpha: ti.f32) -> tm.vec3:
    """Альфа-смешивание двух цветов RGB (top поверх bottom)."""
    return bottom * (1.0 - alpha) + top * alpha


@ti.func
def rot(angle: ti.f32) -> tm.mat2:
    """Матрица поворота 2x2 на заданный угол (против часовой стрелки)."""
    c = ti.cos(angle)
    s = ti.sin(angle)
    return tm.mat2([c, -s], [s, c])


@ti.func
def safe_norm(v: tm.vec2, fallback: tm.vec2) -> tm.vec2:
    """Нормировать вектор, а для почти нулевого вернуть запасное направление."""
    length = v.norm()
    result = fallback
    if length > 1e-6:
        result = v / length
    return result


@ti.func
def smoothmin(a: ti.f32, b: ti.f32, k: ti.f32) -> ti.f32:
    """Полиномиальный сглаживающий минимум (плавное объединение SDF-форм)."""
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


@ti.func
def sd_circle(point: tm.vec2, radius: ti.f32) -> ti.f32:
    """Знаковое расстояние от точки до окружности с центром в начале координат."""
    return point.norm() - radius


@ti.func
def sd_round_cone(p: tm.vec2, a: tm.vec2, b: tm.vec2, r1: ti.f32, r2: ti.f32) -> ti.f32:
    """
    Знаковое расстояние до усеченного конуса (капсулы с разными радиусами концов).

    Это форма отрезка a-b, у которого на конце a радиус r1, а на конце b радиус r2.
    Используется как звено тела рыбы (сглаженная окружность переменного радиуса).

    :param p: точка, для которой считается расстояние
    :param a: первый конец отрезка
    :param b: второй конец отрезка
    :param r1: радиус на конце a
    :param r2: радиус на конце b
    :return: знаковое расстояние (< 0 внутри тела)
    """
    ba = b - a
    l2 = ba.dot(ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2

    pa = p - a
    y = pa.dot(ba)
    z = y - l2
    diff = pa * l2 - ba * y
    x2 = diff.dot(diff)
    y2 = y * y * l2
    z2 = z * z * l2

    k = tm.sign(rr) * rr * rr * x2
    result = (ti.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
    if tm.sign(z) * a2 * z2 > k:
        result = ti.sqrt(x2 + z2) * il2 - r2
    elif tm.sign(y) * a2 * y2 < k:
        result = ti.sqrt(x2 + y2) * il2 - r1
    return result


@ti.data_oriented
class FishShader:
    """
    Шейдер рыбы, двухшаговый: расчет позвоночника + отрисовка.

    Реализованные модификации из списка задания:
      1б. Следование за курсором с запаздыванием (инерция головы).
      2. Ограничение минимального угла между сегментами позвоночника.
      4. Подвижность плавников (взмах во времени + реакция на кривизну).
      5. Цветное отображение существа отдельными слоями со смешиванием.
    """

    def __init__(self, title: str, res: Optional[Tuple[int, int]] = None, gamma: float = 2.2) -> None:
        """
        Задать параметры окна, выделить поля состояния и заполнить начальную позу.

        :param title: заголовок окна
        :param res: разрешение (ширина, высота) в пикселях
        :param gamma: показатель гамма-коррекции (0 - выключить)
        """
        self.title = title
        self.res = res if res is not None else (1000, 562)
        self.resf = tm.vec2(float(self.res[0]), float(self.res[1]))
        self.aspect = float(self.res[0]) / float(self.res[1])
        self.gamma = gamma
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=self.res)

        # Параметры позвоночника
        self.N = 14 # количество узлов
        self.seg_len = 0.052 # длина одного сегмента
        self.body_r = 0.075 # максимальный радиус тела

        # Поля состояния позвоночника
        self.joints = ti.Vector.field(2, dtype=ti.f32, shape=self.N) # рабочая цепочка
        self.disp = ti.Vector.field(2, dtype=ti.f32, shape=self.N) # узлы для отрисовки (с волной)
        self.radii = ti.field(dtype=ti.f32, shape=self.N) # профиль радиусов
        self.head_vel = ti.Vector.field(2, dtype=ti.f32, shape=()) # скорость головы

        # Параметры расчета
        self.dt = 1.0 / 60.0
        self.follow_stiffness = 45.0 # жесткость пружины головы (модификация 1б)
        self.follow_damping = 16.0 # затухание скорости головы
        self.max_turn = 0.45 # макс. угол излома на узле, рад (модификация 2)

        # Параметры волны плавания
        self.wave_amp = 0.024
        self.wave_k = 6.0
        self.wave_w = 6.0

        # Параметры плавников (модификация 4)
        self.fin_joint = 4 # узел крепления грудных плавников
        self.fin_len = 0.115
        self.fin_wid = 0.045
        self.fin_swing = 0.35 # амплитуда взмаха
        self.fin_w = 6.0 # частота взмаха
        self.fin_curv_gain = 1.6 # реакция плавника на кривизну позвоночника
        self.tail_len = 0.105
        self.tail_wid = 0.052
        self.tail_spread = 0.55 # развод лепестков хвоста
        self.tail_swing = 0.22

        # Параметры глаз
        self.eye_fwd = 0.030
        self.eye_side = 0.026
        self.eye_r = 0.018
        self.pupil_r = 0.0085

        self._init_state()

    def _init_state(self) -> None:
        """Заполнить начальную позу позвоночника прямой линией и профиль радиусов."""
        for i in range(self.N):
            self.joints[i] = (-i * self.seg_len, 0.0)
            self.disp[i] = (-i * self.seg_len, 0.0)
            s = i / (self.N - 1)
            bump = math.exp(-((s - 0.27) ** 2) / (2.0 * 0.18 ** 2))
            self.radii[i] = self.body_r * (0.08 + bump)
        self.head_vel[None] = (0.0, 0.0)


    @ti.kernel
    def compute(self, t: ti.f32, cursor: tm.vec2):
        """
        Первый шаг кадра: пересчитать положение узлов позвоночника.

        Этапы: инерционное движение головы к курсору (модификация 1б),
        выравнивание длин сегментов (нерастяжимость), ограничение минимального
        угла между сегментами (модификация 2) и расчет отображаемых узлов с
        бегущей волной плавания.

        :param t: время от запуска, секунды
        :param cursor: позиция курсора в нормализованных координатах [0, 1] x [0, 1]
        """
        # Цель в координатах сцены (центр экрана - начало координат)
        target = tm.vec2((cursor.x - 0.5) * self.aspect, cursor.y - 0.5)

        # Голова как затухающая пружина: ускорение зависит от расстояния до цели
        head = self.joints[0]
        v = self.head_vel[None]
        acc = (target - head) * self.follow_stiffness - v * self.follow_damping
        v = v + acc * self.dt
        head = head + v * self.dt
        self.head_vel[None] = v
        self.joints[0] = head

        # Нерастяжимость: каждый узел держится на seg_len от предыдущего
        ti.loop_config(serialize=True)
        for i in range(1, self.N):
            prev = self.joints[i - 1]
            direction = safe_norm(self.joints[i] - prev, tm.vec2(-1.0, 0.0))
            self.joints[i] = prev + direction * self.seg_len

        # Ограничение минимального угла между соседними сегментами (модификация 2)
        ti.loop_config(serialize=True)
        for i in range(1, self.N - 1):
            a = self.joints[i - 1]
            b = self.joints[i]
            c = self.joints[i + 1]
            u1 = safe_norm(b - a, tm.vec2(-1.0, 0.0))
            u2 = safe_norm(c - b, tm.vec2(-1.0, 0.0))
            cos_turn = u1.dot(u2)
            # излом слишком резкий, если угол поворота больше max_turn
            if cos_turn < ti.cos(self.max_turn):
                cross = u1.x * u2.y - u1.y * u2.x
                sign = 1.0 if cross >= 0.0 else -1.0
                new_dir = rot(sign * self.max_turn) @ u1
                self.joints[i + 1] = b + new_dir * self.seg_len

        # Отображаемые узлы: добавляем бегущую волну поперек тела
        ti.loop_config(serialize=True)
        for i in range(self.N):
            prev_i = max(i - 1, 0)
            next_i = min(i + 1, self.N - 1)
            tangent = safe_norm(self.joints[next_i] - self.joints[prev_i], tm.vec2(1.0, 0.0))
            perp = tm.vec2(-tangent.y, tangent.x)
            s = i / (self.N - 1)
            amp = self.wave_amp * ti.pow(s, 1.4)
            phase = self.wave_k * s - self.wave_w * t
            self.disp[i] = self.joints[i] + perp * amp * ti.sin(phase)


    @ti.func
    def body_sdf(self, p: tm.vec2) -> ti.f32:
        """Расстояние до тела рыбы (сглаженное объединение звеньев)."""
        d = 1.0e9
        for i in range(self.N - 1):
            link = sd_round_cone(p, self.disp[i], self.disp[i + 1], self.radii[i], self.radii[i + 1])
            d = smoothmin(d, link, 0.02)
        return d

    @ti.func
    def fin_alpha(self, p: tm.vec2, anchor: tm.vec2, direction: tm.vec2,
                  length: ti.f32, width: ti.f32, aa: ti.f32) -> ti.f32:
        """
        Покрытие пикселя эллиптическим плавником.

        Плавник - эллипс с центром, сдвинутым от точки крепления вдоль
        direction, вытянутый по direction и узкий поперек.

        :param p: точка (координаты сцены)
        :param anchor: точка крепления плавника к телу
        :param direction: единичное направление плавника
        :param length: длина плавника
        :param width: ширина плавника
        :param aa: ширина сглаживания края
        :return: покрытие [0, 1]
        """
        center = anchor + direction * (length * 0.5)
        angle = ti.atan2(direction.y, direction.x)
        local = rot(-angle) @ (p - center)
        local.x /= length * 0.5
        local.y /= width * 0.5
        d = (local.norm() - 1.0) * min(length, width) * 0.5
        return alpha_from_sdf(d, aa)

    @ti.func
    def main_image(self, uv: tm.vec2, t: ti.f32) -> tm.vec3:
        """
        Второй шаг кадра: цвет одного пикселя как смешивание слоев (модификация 5).

        :param uv: координаты пикселя (центр экрана - начало координат)
        :param t: время от запуска
        :return: цвет пикселя RGB в линейном пространстве
        """
        aa = 1.2 / self.resf.y
        p = uv

        # Слой 0: фон
        water_top = tm.vec3(0.10, 0.42, 0.55)
        water_bottom = tm.vec3(0.03, 0.13, 0.26)
        color = blend(water_bottom, water_top, smooth(-0.5, 0.5, uv.y))

        # Направления на ключевых узлах
        head = self.disp[0]
        head_dir = safe_norm(self.disp[0] - self.disp[1], tm.vec2(-1.0, 0.0))

        # Слой 1: хвостовой плавник (два лепестка) - рисуем первым, тело его перекроет
        tail = self.disp[self.N - 1]
        tail_dir = safe_norm(self.disp[self.N - 1] - self.disp[self.N - 2], tm.vec2(1.0, 0.0))
        tail_osc = self.tail_swing * ti.sin(t * self.fin_w + 0.6)
        up_dir = rot(self.tail_spread + tail_osc) @ tail_dir
        dn_dir = rot(-self.tail_spread + tail_osc) @ tail_dir
        tail_a = self.fin_alpha(p, tail, up_dir, self.tail_len, self.tail_wid, aa)
        tail_a = max(tail_a, self.fin_alpha(p, tail, dn_dir, self.tail_len, self.tail_wid, aa))
        tail_color = tm.vec3(0.90, 0.40, 0.12)
        color = blend(color, tail_color, tail_a * 0.92)

        # Слой 2: грудные плавники (пара конечностей, модификация 4)
        af = self.fin_joint
        axis = safe_norm(self.disp[af - 1] - self.disp[af + 1], tm.vec2(-1.0, 0.0))
        perp = tm.vec2(-axis.y, axis.x)
        back = -axis
        # кривизна позвоночника в точке крепления
        d1 = safe_norm(self.disp[af] - self.disp[af - 1], tm.vec2(1.0, 0.0))
        d2 = safe_norm(self.disp[af + 1] - self.disp[af], tm.vec2(1.0, 0.0))
        curv = d1.x * d2.y - d1.y * d2.x
        flap = self.fin_swing * ti.sin(t * self.fin_w) + self.fin_curv_gain * curv
        fin_color = tm.vec3(0.98, 0.62, 0.26)
        for side in ti.static(range(2)):
            sgn = 1.0 if side == 0 else -1.0
            base_dir = safe_norm(back * 0.5 + perp * sgn, perp * sgn)
            fin_dir = rot(sgn * flap) @ base_dir
            anchor = self.disp[af] + perp * sgn * (self.radii[af] * 0.5)
            fa = self.fin_alpha(p, anchor, fin_dir, self.fin_len, self.fin_wid, aa)
            color = blend(color, fin_color, fa * 0.85)

        # Слой 3: тело
        body_a = alpha_from_sdf(self.body_sdf(p), aa)
        body_color = tm.vec3(0.93, 0.50, 0.16)
        # легкое затемнение спины (верх кадра) для объема
        body_color = body_color * (0.82 + 0.18 * smooth(0.5, -0.5, uv.y))
        color = blend(color, body_color, body_a)

        # Слой 4: глаза (пара) с зрачком и бликом
        eye_perp = tm.vec2(-head_dir.y, head_dir.x)
        for side in ti.static(range(2)):
            sgn = 1.0 if side == 0 else -1.0
            eye_c = head + head_dir * self.eye_fwd + eye_perp * sgn * self.eye_side
            white_a = alpha_from_sdf(sd_circle(p - eye_c, self.eye_r), aa)
            color = blend(color, tm.vec3(0.96, 0.96, 0.94), white_a)
            pupil_c = eye_c + head_dir * 0.004
            pupil_a = alpha_from_sdf(sd_circle(p - pupil_c, self.pupil_r), aa)
            color = blend(color, tm.vec3(0.05, 0.05, 0.07), pupil_a)
            glint_a = alpha_from_sdf(sd_circle(p - (pupil_c + eye_perp * sgn * 0.003 + head_dir * 0.003), 0.003), aa)
            color = blend(color, tm.vec3(1.0, 1.0, 1.0), glint_a)

        return tm.clamp(color, 0.0, 1.0)


    @ti.kernel
    def render(self, t: ti.f32):
        """
        Второй шаг кадра: заполнить буфер пикселей цветами (параллельно по пикселям).

        :param t: время от запуска, секунды
        """
        for frag in ti.grouped(self.pixels):
            uv = (frag - 0.5 * self.resf) / self.resf.y
            color = self.main_image(uv, t)
            if self.gamma > 0.0:
                color = tm.clamp(color ** (1.0 / self.gamma), 0.0, 1.0)
            self.pixels[frag] = color


    def main_loop(self) -> None:
        """Открыть окно и крутить цикл: на каждом кадре сначала compute, потом render."""
        gui = ti.GUI(self.title, res=self.res, fast_gui=True)
        start = time.time()

        while gui.running:
            if gui.get_event(ti.GUI.PRESS) and gui.event.key == ti.GUI.ESCAPE:
                break

            t = time.time() - start
            cursor = tm.vec2(*gui.get_cursor_pos())
            self.compute(t, cursor)
            self.render(t)
            gui.set_image(self.pixels)
            gui.show()

        gui.close()


def main() -> None:
    """Инициализировать Taichi и запустить шейдер рыбы."""
    try:
        ti.init(arch=ti.gpu, offline_cache=False)
    except Exception:
        ti.init(arch=ti.cpu, offline_cache=False)

    shader = FishShader(title="shaders3_Pechenin", res=(1000, 562), gamma=2.2)
    shader.main_loop()


if __name__ == "__main__":
    main()
