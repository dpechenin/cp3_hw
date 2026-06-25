import taichi as ti


def init_taichi():
    """
    Запуск Taichi на GPU, при неудаче откат на CPU.
    Возвращает имя выбранного backend.
    """
    try:
        ti.init(arch=ti.gpu)
        return str(ti.lang.impl.current_cfg().arch)
    except Exception:
        ti.init(arch=ti.cpu)
        return str(ti.lang.impl.current_cfg().arch)


# Физический размер области
WIDTH = 90.0
HEIGHT = 60.0


@ti.func
def smoothstep(edge0, edge1, x):
    """Плавный переход от 0 до 1 между edge0 и edge1."""
    t = ti.math.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def sd_box(px, py, cx, cy, hx, hy):
    """SDF прямоугольника с центром (cx,cy) и полуразмерами (hx,hy)."""
    dx = ti.abs(px - cx) - hx
    dy = ti.abs(py - cy) - hy
    ax = ti.max(dx, 0.0)
    ay = ti.max(dy, 0.0)
    return ti.sqrt(ax * ax + ay * ay) + ti.min(ti.max(dx, dy), 0.0)


@ti.func
def sd_circle(px, py, cx, cy, r):
    """SDF круга с центром (cx,cy) и радиусом r."""
    return ti.sqrt((px - cx) ** 2 + (py - cy) ** 2) - r


@ti.func
def sd_lens(px, py):
    """
    Двояковогнутая линза: прямоугольник, из которого вырезаны
    два круга слева и справа.
    """
    box = sd_box(px, py, 27.5, 25.0, 7.5, 15.0)
    circ_l = sd_circle(px, py, 5.0, 25.0, 20.0)
    circ_r = sd_circle(px, py, 50.0, 25.0, 20.0)
    return ti.max(ti.max(box, -circ_l), -circ_r)


@ti.data_oriented
class WaveSolver:
    def __init__(self, nx=900, ny=600, base_kappa=1.0 / 1.5,
                 n=(1.30, 1.35, 1.40), acc=0.15):
        self.nx = nx
        self.ny = ny
        self.base_kappa = base_kappa
        self.acc = acc

        # u[цвет, время(0 будущее, 1 настоящее, 2 прошлое), y, x]
        self.u = ti.field(ti.f32, shape=(3, 3, ny, nx))
        self.kappa = ti.field(ti.f32, shape=(3, ny, nx))
        self.accum = ti.field(ti.f32, shape=(3, ny, nx))
        self.mirror = ti.field(ti.i32, shape=(ny, nx))
        self.tmask = ti.field(ti.f32, shape=(ny, nx))
        self.n = ti.field(ti.f32, shape=3)
        for c in range(3):
            self.n[c] = n[c]

    @ti.func
    def phys(self, i, j):
        """Перевод индексов пикселя (i по y, j по x) в физические координаты."""
        x = (j + 0.5) / self.nx * WIDTH
        y = (i + 0.5) / self.ny * HEIGHT
        return x, y

    @ti.kernel
    def build_geometry(self):
        # Линза: гладкая маска и замедление волн внутри стекла
        for i, j in ti.ndrange(self.ny, self.nx):
            x, y = self.phys(i, j)
            d = sd_lens(x, y)
            # сглаживание на ширину ~0.15 единицы
            m = smoothstep(0.15, -0.15, d)
            self.tmask[i, j] = m
            for c in range(3):
                self.kappa[c, i, j] = self.base_kappa * (m / self.n[c] + (1.0 - m))
        # Зеркало: тонкий горизонтальный отрезок (условия Дирихле)
        for i, j in ti.ndrange(self.ny, self.nx):
            x, y = self.phys(i, j)
            if ti.abs(y - 45.0) <= 0.3 and 40.0 <= x <= 60.0:
                self.mirror[i, j] = 1

    @ti.kernel
    def init_impulse(self, xs: ti.f32, ys: ti.f32, angle: ti.f32,
                     freq: ti.f32, sigma_s: ti.f32, sigma_p: ti.f32):
        """Направленный импульс луча с нулевой начальной скоростью."""
        ca = ti.cos(angle)
        sa = ti.sin(angle)
        for i, j in ti.ndrange(self.ny, self.nx):
            x, y = self.phys(i, j)
            dx = x - xs
            dy = y - ys
            # s - координата вдоль луча, p - поперек.
            s = dx * ca + dy * sa
            p = -dx * sa + dy * ca
            val = ti.exp(-0.5 * ((s / sigma_s) ** 2 + (p / sigma_p) ** 2)) \
                * ti.cos(freq * s)
            for c in range(3):
                self.u[c, 0, i, j] = val
                self.u[c, 1, i, j] = val
                self.u[c, 2, i, j] = val

    @ti.kernel
    def open_boundary(self):
        """Открытая граница (условия Мура 1-го порядка) на 4 краях области."""
        # верхняя и нижняя границы (y = 0 и y = ny-1)
        for c, j in ti.ndrange(3, self.nx):
            k0 = self.kappa[c, 0, j]
            self.u[c, 0, 0, j] = self.u[c, 1, 1, j] \
                + (k0 - 1.0) / (k0 + 1.0) * (self.u[c, 0, 1, j] - self.u[c, 1, 0, j])
            kn = self.kappa[c, self.ny - 1, j]
            self.u[c, 0, self.ny - 1, j] = self.u[c, 1, self.ny - 2, j] \
                + (kn - 1.0) / (kn + 1.0) \
                * (self.u[c, 0, self.ny - 2, j] - self.u[c, 1, self.ny - 1, j])
        # левая и правая границы (x = 0 и x = nx-1)
        for c, i in ti.ndrange(3, self.ny):
            k0 = self.kappa[c, i, 0]
            self.u[c, 0, i, 0] = self.u[c, 1, i, 1] \
                + (k0 - 1.0) / (k0 + 1.0) * (self.u[c, 0, i, 1] - self.u[c, 1, i, 0])
            kn = self.kappa[c, i, self.nx - 1]
            self.u[c, 0, i, self.nx - 1] = self.u[c, 1, i, self.nx - 2] \
                + (kn - 1.0) / (kn + 1.0) \
                * (self.u[c, 0, i, self.nx - 2] - self.u[c, 1, i, self.nx - 1])

    @ti.kernel
    def step_shift(self):
        """Сдвиг слоев по времени: настоящее в прошлое, будущее в настоящее."""
        for c, i, j in ti.ndrange(3, self.ny, self.nx):
            self.u[c, 2, i, j] = self.u[c, 1, i, j]
            self.u[c, 1, i, j] = self.u[c, 0, i, j]

    @ti.kernel
    def propagate(self):
        """Решеточное уравнение: шаг по времени для всех внутренних узлов."""
        for c, i, j in ti.ndrange(3, (1, self.ny - 1), (1, self.nx - 1)):
            k = self.kappa[c, i, j]
            lap = (self.u[c, 1, i - 1, j] + self.u[c, 1, i + 1, j]
                   + self.u[c, 1, i, j - 1] + self.u[c, 1, i, j + 1]
                   - 4.0 * self.u[c, 1, i, j])
            self.u[c, 0, i, j] = k * k * lap \
                + 2.0 * self.u[c, 1, i, j] - self.u[c, 2, i, j]

    @ti.kernel
    def mirror_dirichlet(self):
        """Условие Дирихле на зеркале: зануляем поле в его узлах."""
        for c, i, j in ti.ndrange(3, self.ny, self.nx):
            if self.mirror[i, j] == 1:
                self.u[c, 0, i, j] = 0.0

    @ti.kernel
    def accumulate(self):
        """Копим модуль амплитуды."""
        for c, i, j in ti.ndrange(3, (1, self.ny - 1), (1, self.nx - 1)):
            self.accum[c, i, j] += self.acc * ti.abs(self.u[c, 0, i, j]) \
                * self.kappa[c, i, j] / self.base_kappa

    @ti.kernel
    def reset_fields(self):
        """Обнуление полей перед повторным прогоном (геометрия сохраняется)."""
        for c, t, i, j in ti.ndrange(3, 3, self.ny, self.nx):
            self.u[c, t, i, j] = 0.0
        for c, i, j in ti.ndrange(3, self.ny, self.nx):
            self.accum[c, i, j] = 0.0

    def step(self):
        """Один шаг расчета."""
        self.open_boundary()
        self.step_shift()
        self.propagate()
        self.mirror_dirichlet()
        self.accumulate()
