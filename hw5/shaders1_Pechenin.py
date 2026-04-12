"""
Вариант 02 (часы).

1. Элементы анимации
1.1. Фон
Кадр заполнен вертикальным градиентом: сверху светло-голубой, снизу сине-серый.
Дополнительных объектов на фоне нет.

1.2. Циферблат
В центре кадра расположен большой круглый циферблат с почти белой заливкой.
Тон практически однородный, лишь у края чуть темнее. По внешнему контуру идет толстая темная окантовка.

1.3. Деления
На циферблате есть 60 делений. Каждое деление ориентировано по радиусу.
12 часовых делений длиннее и толще, остальные 48 минутных делений тоньше и короче.

1.4. Стрелки
В сцене три стрелки:
- длинная черная минутная стрелка (в начале направлена 43 минуты)
- короткая черная часовая стрелка (в начале направлена на 6 часов)
- тонкая красная секундная стрелка (в начале направлена на 27 секунду)
Все стрелки перемещаются дискретно. Секундная делает один шаг в секунду.
Минутная делает один шаг в тот момент, когда секундная проходит отметку 12 часов.
Часовая делает один шаг в тот момент, когда минутная проходит отметку 12 часов.

1.5. Центральная накладка
В месте соединения стрелок находится небольшая круглая ось серого цвета с темной
обводкой. Она закрывает основания стрелок.

1.6. Тени
У стрелок есть мягкие смещенные тени. Они направлены так, как будто свет светит
примерно сверху справа.

2. Функции и ссылки на Graphtoy
Ниже независимая переменная x используется как удобная замена для соответствующей
величины из шейдера: времени t, вертикальной координаты y или радиуса r.

2.1. Вертикальный градиент фона
Формула: g(y) = smoothstep(-0.55, 0.55, y)
Ссылка:
https://graphtoy.com/?f1(x,t)=smoothstep(-0.55,0.55,x)&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=&v4=false&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=0,0,3.2

2.2. Радиальное затемнение циферблата
Формула: h(r) = smoothstep(0.08, 0.48, r)
Ссылка:
https://graphtoy.com/?f1(x,t)=smoothstep(0.08,0.48,abs(x))&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=&v4=false&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=0,0,3.2

2.3. Угол секундной стрелки
Формула: a_s(t) = -72 - 6 * floor(t) градусов
Ссылка:
https://graphtoy.com/?f1(x,t)=-72-6*floor(x)&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=&v4=false&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=0,0,40

2.4. Угол минутной стрелки
Формула: a_m(t) = -168 - 6 * floor((27 + floor(t)) / 60) градусов
Ссылка:
https://graphtoy.com/?f1(x,t)=-168-6*floor((27+floor(x))/60)&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=&v4=false&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=0,0,220

2.5. Угол часовой стрелки
Формула: a_h(t) = -90 - 30 * floor((43 + floor((27 + floor(t)) / 60)) / 60) градусов
Ссылка:
https://graphtoy.com/?f1(x,t)=-90-30*floor((43+floor((27+floor(x))/60))/60)&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=&v4=false&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=0,0,160

3. Взаимосвязи элементов
Сначала строится фон. Затем поверх него рисуется круг циферблата, после чего
добавляются деления. Следующим слоем идут тени стрелок, затем сами стрелки,
и в самом конце - центральная ось. Динамика кадра создается дискретной
зависимостью углов стрелок от времени: каждая стрелка меняет положение только
в момент очередного такта, а между тактами остается неподвижной.

4. Идея реализации
Сцена удобно описывается через signed distance functions:
- круг циферблата задается SDF окружности
- окантовка задается SDF кольца
- стрелки и деления задаются SDF отрезков с закругленными концами
- 60 и 12 делений строятся не отдельными ручными объектами, а через повторение
  пространства по углу, чтобы переиспользовать одну и ту же геометрию
- мягкие края получаются через сглаженный переход по расстоянию
"""

from typing import Optional, Tuple
import math
import time

import taichi as ti
import taichi.math as tm

@ti.func
def smooth(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    """Return a smooth Hermite interpolation between two scalar edges."""
    h = tm.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return h * h * (3.0 - 2.0 * h)


@ti.func
def alpha_from_sdf(distance: ti.f32, aa: ti.f32) -> ti.f32:
    """Convert a signed distance to anti-aliased coverage."""
    return 1.0 - smooth(-aa, aa, distance)


@ti.func
def blend(bottom: tm.vec3, top: tm.vec3, alpha: ti.f32) -> tm.vec3:
    """Alpha-blend two RGB colors."""
    return bottom * (1.0 - alpha) + top * alpha


@ti.func
def rot(angle: ti.f32) -> tm.mat2:
    """Build a 2D rotation matrix."""
    c = ti.cos(angle)
    s = ti.sin(angle)
    return tm.mat2([c, -s], [s, c])


@ti.func
def sd_circle(point: tm.vec2, radius: ti.f32) -> ti.f32:
    """Return signed distance from a point to a circle."""
    return tm.length(point) - radius


@ti.func
def d_segment(point: tm.vec2, start: tm.vec2, end: tm.vec2) -> ti.f32:
    """Return unsigned distance from a point to a finite segment."""
    pa = point - start
    ba = end - start
    h = tm.clamp((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return tm.length(pa - ba * h)


@ti.func
def sd_capsule(point: tm.vec2, start: tm.vec2, end: tm.vec2, radius: ti.f32) -> ti.f32:
    """Return signed distance from a point to a rounded segment."""
    return d_segment(point, start, end) - radius


@ti.func
def sd_ring(point: tm.vec2, radius: ti.f32, half_width: ti.f32) -> ti.f32:
    """Return signed distance from a point to a ring."""
    return ti.abs(tm.length(point) - radius) - half_width


@ti.func
def angular_repeat(point: tm.vec2, sector: ti.f32) -> tm.vec2:
    """Rotate a point into the local coordinate system of the nearest angular sector."""
    angle = ti.atan2(point.y, point.x)
    cell = ti.floor((angle + 0.5 * sector) / sector)
    return rot(-cell * sector) @ point


@ti.data_oriented
class BaseShader:
    """Taichi shader base class."""

    def __init__(self, title: str, res: Optional[Tuple[int, int]] = None, gamma: float = 2.2,) -> None:
        """Store window parameters and allocate the framebuffer."""
        self.title = title
        self.res = res if res is not None else (1000, 562)
        self.resf = tm.vec2(float(self.res[0]), float(self.res[1]))
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=self.res)
        self.gamma = gamma

    @ti.func
    def main_image(self, uv: tm.vec2, t: ti.f32, cursor: tm.vec2) -> tm.vec3:
        """Return the color of a single pixel in normalized coordinates."""
        color = tm.vec3(0.0)
        color.rg = uv + 0.5
        return color

    @ti.kernel
    def render(self, t: ti.f32, cursor: tm.vec2):
        """Render one frame into the framebuffer."""
        for frag_coord in ti.grouped(self.pixels):
            uv = (frag_coord - 0.5 * self.resf) / self.resf.y
            color = self.main_image(uv, t, cursor)
            if self.gamma > 0.0:
                color = tm.clamp(color ** (1.0 / self.gamma), 0.0, 1.0)
            self.pixels[frag_coord] = color

    def main_loop(self) -> None:
        """Open the GUI window and render frames until the user closes it."""
        gui = ti.GUI(self.title, res=self.res, fast_gui=False)
        start = time.time()

        while gui.running:
            if gui.get_event(ti.GUI.PRESS) and gui.event.key == ti.GUI.ESCAPE:
                break

            elapsed = time.time() - start
            cursor = gui.get_cursor_pos()
            self.render(elapsed, cursor)
            gui.set_image(self.pixels)
            gui.show()

        gui.close()


@ti.data_oriented
class ClockShader(BaseShader):
    """Recreate the animated clock from variant 02 with analytic SDF geometry."""

    def __init__(self) -> None:
        """Set constants that control the clock layout and animation."""
        super().__init__("shaders1_Pechenin", res=(1000, 562), gamma=2.2)

        self.clock_radius = 0.481
        self.ring_half_width = 0.0110

        self.tick_outer = 0.470
        self.minor_tick_inner = 0.438
        self.major_tick_inner = 0.404
        self.minor_tick_radius = 0.0016
        self.major_tick_radius = 0.0028

        self.long_hand_start = -168.0 * math.pi / 180.0
        self.long_hand_step = 6.0 * math.pi / 180.0
        self.long_hand_length = 0.360
        self.long_hand_radius = 0.0068

        self.short_hand_start = -90.0 * math.pi / 180.0
        self.short_hand_step = 30.0 * math.pi / 180.0
        self.short_hand_length = 0.205
        self.short_hand_radius = 0.0082

        self.second_hand_start = -72.0 * math.pi / 180.0
        self.second_hand_speed = 6.0 * math.pi / 180.0
        self.second_hand_length = 0.454
        self.second_hand_radius = 0.0033

        self.second_start_index = self.clock_index_from_angle(self.second_hand_start, 60)
        self.minute_start_index = self.clock_index_from_angle(self.long_hand_start, 60)
        self.hour_start_index = self.clock_index_from_angle(self.short_hand_start, 12)

        self.shadow_offset = tm.vec2(-0.022, -0.026)

    @staticmethod
    def clock_index_from_angle(angle: float, divisions: int) -> float:
        """Convert an initial hand angle to the matching clock tick index."""
        step = 2.0 * math.pi / divisions
        index = round((0.5 * math.pi - angle) / step)
        return float(index % divisions)

    @ti.func
    def background_color(self, uv: tm.vec2) -> tm.vec3:
        """Return the blue vertical background gradient, light blue at top fading to blue-grey at bottom."""
        top = tm.vec3(0.68, 0.78, 0.89)
        bottom = tm.vec3(0.12, 0.20, 0.32)
        mix_value = smooth(-0.55, 0.55, uv.y)
        color = blend(bottom, top, mix_value)
        return tm.clamp(color, 0.0, 1.0)

    @ti.func
    def face_color(self, point: tm.vec2) -> tm.vec3:
        """Return the nearly uniform white dial color with subtle radial darkening toward the edge."""
        radius = tm.length(point)
        radial_shade = smooth(0.08, 0.48, radius)

        base = tm.vec3(0.920, 0.915, 0.910)
        base -= tm.vec3(0.24, 0.24, 0.23) * radial_shade
        return tm.clamp(base, 0.0, 1.0)

    @ti.func
    def hand_end(self, angle: ti.f32, length: ti.f32) -> tm.vec2:
        """Return the endpoint of a hand given its angle and length."""
        return tm.vec2(ti.cos(angle), ti.sin(angle)) * length

    @ti.func
    def second_hand_wraps(self, t: ti.f32) -> ti.f32:
        """Return how many times the second hand has crossed 12 o'clock."""
        whole_seconds = ti.floor(t)
        return ti.floor((self.second_start_index + whole_seconds) / 60.0)

    @ti.func
    def minute_hand_wraps(self, t: ti.f32) -> ti.f32:
        """Return how many times the minute hand has crossed 12 o'clock."""
        wraps = self.second_hand_wraps(t)
        return ti.floor((self.minute_start_index + wraps) / 60.0)

    @ti.func
    def second_hand_angle(self, t: ti.f32) -> ti.f32:
        """Return the animated angle of the red second hand."""
        return self.second_hand_start - self.second_hand_speed * ti.floor(t)

    @ti.func
    def minute_hand_angle(self, t: ti.f32) -> ti.f32:
        """Return the stepped angle of the minute hand."""
        return self.long_hand_start - self.long_hand_step * self.second_hand_wraps(t)

    @ti.func
    def hour_hand_angle(self, t: ti.f32) -> ti.f32:
        """Return the stepped angle of the hour hand."""
        return self.short_hand_start - self.short_hand_step * self.minute_hand_wraps(t)

    @ti.func
    def tick_alpha(self, point: tm.vec2, aa: ti.f32) -> ti.f32:
        """Return combined coverage of minor and major tick marks."""
        minor_point = angular_repeat(point, 2.0 * math.pi / 60.0)
        major_point = angular_repeat(point, 2.0 * math.pi / 12.0)

        minor_distance = sd_capsule(
            minor_point,
            tm.vec2(self.minor_tick_inner, 0.0),
            tm.vec2(self.tick_outer, 0.0),
            self.minor_tick_radius,
        )
        major_distance = sd_capsule(
            major_point,
            tm.vec2(self.major_tick_inner, 0.0),
            tm.vec2(self.tick_outer, 0.0),
            self.major_tick_radius,
        )

        minor_alpha = alpha_from_sdf(minor_distance, aa)
        major_alpha = alpha_from_sdf(major_distance, aa)
        return tm.clamp(minor_alpha + major_alpha, 0.0, 1.0)

    @ti.func
    def hand_alpha(self, point: tm.vec2, angle: ti.f32, length: ti.f32, radius: ti.f32, aa: ti.f32) -> ti.f32:
        """Return coverage of a rounded hand segment."""
        return alpha_from_sdf(sd_capsule(point, tm.vec2(0.0), self.hand_end(angle, length), radius), aa)

    @ti.func
    def hand_shadow_alpha(self, point: tm.vec2, angle: ti.f32, length: ti.f32, radius: ti.f32, aa: ti.f32) -> ti.f32:
        """Return coverage of the soft shadow of a hand."""
        shadow_point = point - self.shadow_offset
        shadow_radius = radius * 1.28 + 0.0015
        return alpha_from_sdf(
            sd_capsule(
                shadow_point,
                tm.vec2(0.0),
                self.hand_end(angle, length),
                shadow_radius,
            ),
            aa * 3.40,
        )

    @ti.func
    def main_image(self, uv: tm.vec2, t: ti.f32, cursor: tm.vec2) -> tm.vec3:
        """Render the clock image in normalized coordinates."""
        aa = 0.7 / self.resf.y
        point = uv
        radius = tm.length(point)

        color = self.background_color(point)

        face_alpha = alpha_from_sdf(sd_circle(point, self.clock_radius), aa * 0.9)
        color = blend(color, self.face_color(point), face_alpha)

        tick_color = tm.vec3(0.01, 0.01, 0.01)
        color = blend(color, tick_color, self.tick_alpha(point, aa))

        ring_alpha = alpha_from_sdf(sd_ring(point, self.clock_radius, self.ring_half_width), aa)
        color = blend(color, tm.vec3(0.0, 0.0, 0.0), ring_alpha)

        long_angle = self.minute_hand_angle(t)
        short_angle = self.hour_hand_angle(t)
        second_angle = self.second_hand_angle(t)

        long_shadow = self.hand_shadow_alpha(point, long_angle, self.long_hand_length, self.long_hand_radius, aa)
        short_shadow = self.hand_shadow_alpha(point, short_angle, self.short_hand_length, self.short_hand_radius, aa)
        second_shadow = self.hand_shadow_alpha(point, second_angle, self.second_hand_length, self.second_hand_radius, aa)

        shadow_color = tm.vec3(0.10, 0.10, 0.11)
        color = blend(color, shadow_color, long_shadow * 0.20)
        color = blend(color, shadow_color, short_shadow * 0.20)
        color = blend(color, shadow_color, second_shadow * 0.14)

        hand_dark = tm.vec3(0.0, 0.0, 0.0)
        long_alpha = self.hand_alpha(point, long_angle, self.long_hand_length, self.long_hand_radius, aa)
        short_alpha = self.hand_alpha(point, short_angle, self.short_hand_length, self.short_hand_radius, aa)
        second_alpha = self.hand_alpha(point, second_angle, self.second_hand_length, self.second_hand_radius, aa)

        color = blend(color, hand_dark, long_alpha)
        color = blend(color, hand_dark, short_alpha)
        color = blend(color, tm.vec3(0.62, 0.04, 0.06), second_alpha)

        axis_shadow = alpha_from_sdf(sd_circle(point - self.shadow_offset, 0.0285), aa * 3.2)
        color = blend(color, shadow_color, axis_shadow * 0.15)

        axis_fill = alpha_from_sdf(sd_circle(point, 0.0245), aa)
        axis_outline = alpha_from_sdf(sd_ring(point, 0.0245, 0.0040), aa)
        color = blend(color, tm.vec3(0.54, 0.54, 0.55), axis_fill)
        color = blend(color, tm.vec3(0.0, 0.0, 0.0), axis_outline)

        return tm.clamp(color, 0.0, 1.0)


def main() -> None:
    """Initialize Taichi and run the clock shader."""
    try:
        ti.init(arch=ti.gpu, offline_cache=False)
    except Exception:
        ti.init(arch=ti.cpu, offline_cache=False)

    shader = ClockShader()
    shader.main_loop()


if __name__ == "__main__":
    main()
