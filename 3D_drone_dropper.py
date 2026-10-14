from dataclasses import dataclass, replace
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# ---------- Launch Conditions --------------- 
@dataclass(frozen = True)
class Params:
    speed: float = 15.0       # launch speed, m/s
    elevation_deg: float = 0.0  #xy angle
    azimuth_deg: float = 0.0   #xz angle 
    Cd: float = 0.3         # drag coefficient
    A: float = 0.05         # cross-sectional area, m^2
    m: float = 4.0          # mass, kg
    wind_x: float = 0.0     # wind speed m/s in  x-dir (positive = tailwind)
    wind_y: float = 0.0     # wind speed m/s in  y-dir (positive = updraft)
    wind_z: float = 0.0     # wind speed m/s in  z-dir (positive = tailwind)
    g: float = 9.81         #gravity, m/s^2
    y0: float = 50.0         #release height m



def initial_velocity(p: Params):
    elevation_theta = np.radians(p.elevation_deg)
    azimuth_theta = np.radians(p.azimuth_deg)
    vx_int, vy_int, vz_int = [p.speed * np.cos(elevation_theta) * np.cos(azimuth_theta), p.speed * np.sin(elevation_theta),
                              p.speed*np.sin(azimuth_theta) * np.cos(elevation_theta)]
    return vx_int, vy_int, vz_int

#------------ ODE Solver-----------------

def simulate(p: Params):
    vx0, vy0, vz0 = initial_velocity(p)
    initial_cond = [0.0, p.y0, 0.0, vx0, vy0, vz0]
    def flight_dynamics(t, cond):
        x, y, z, vx, vy, vz = cond
        rho = 1.225 * np.exp(-y/8500)
        vx_rel = vx - p.wind_x
        vy_rel = vy - p.wind_y
        vz_rel = vz - p.wind_z
        v_rel = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)
        F_drag = 0.5 * rho * v_rel**2 * p.Cd * p.A

        if v_rel > 0:
            ax = -(F_drag/p.m) * (vx_rel/v_rel)
            ay = -p.g - (F_drag/p.m) * (vy_rel/v_rel) 
            az = -(F_drag/p.m) * (vz_rel/v_rel)
        else:
            ax = 0
            ay = -p.g
            az = 0

        return [vx, vy, vz, ax, ay, az]
    def hit_ground(t, cond):
        return cond[1]

    hit_ground.terminal = True
    hit_ground.direction = -1
    sol = solve_ivp(flight_dynamics,(0, 300), y0=initial_cond,
            events=hit_ground, dense_output= True, rtol=1e-8, atol=1e-10)
    if sol.status !=  1:
        raise RuntimeError("Projectile never landed within the time frame")
    return sol
def solve_summarize(sol):
    return{"range": sol.y[0, -1], "max_height": sol.y[1].max(), "crossrange": sol.y[2,-1],"time": sol.t[-1], 
           "impact_speed": np.linalg.norm(sol.y[3:6, -1])}


# ------------ Vacuum Comparison ----------------
def vacuum_landing(p: Params):
    vx0, vy0, vz0 = initial_velocity(p)
    t = (vy0 + np.sqrt(vy0**2 + 2 * p.g * p.y0)) / p.g
    return vx0 * t, vz0 * t


def check_against_vacuum():
    p = replace(Params(Cd=0.0), elevation_deg=20.0, azimuth_deg=30.0)
    summary = solve_summarize(simulate(p))
    num_x, num_z = summary["range"], summary["crossrange"]
    ana_x, ana_z = vacuum_landing(p)

    ok_x = np.isclose(num_x, ana_x, rtol=1e-4)
    ok_z = np.isclose(num_z, ana_z, rtol=1e-4, atol=1e-6)
    if not (ok_x and ok_z):
        raise RuntimeError(
            f"Vacuum check FAILED: x numeric {num_x:.6f} vs analytic {ana_x:.6f} m, "
            f"z numeric {num_z:.6f} vs analytic {ana_z:.6f} m")
def check_planar_symmetry():
    p = replace(Params(), azimuth_deg = 0.0, elevation_deg = 35.0, wind_x = 10.0, wind_y = 10.0, wind_z = 0.0)
    sol = simulate(p)
    z_history = sol.y[2]
    worst_drift = np.max(np.abs(z_history))
    if  worst_drift > 1e-12:
        raise RuntimeError(f"Drift check FAILED:absolute largest drift was {worst_drift:.3e} m")
#------------ Wind Functions -------------------
def check_crosswind_sign():
    base = replace(Params(), elevation_deg = 0, azimuth_deg = 0, wind_x = 0, wind_y = 0)
    cross_range_pos = solve_summarize(simulate(replace(base, wind_z = 5.0)))["crossrange"]
    cross_range_neg = solve_summarize(simulate(replace(base, wind_z = -5.0)))["crossrange"]
    total = cross_range_pos + cross_range_neg
    if cross_range_pos <= 0:
        raise ValueError(f"+5 m/s crosswind gave crossrange {cross_range_pos:.3e} m, expected > 0")
    elif cross_range_neg >= 0:
        raise ValueError(f"-5 m/s crosswind gave crossrange {cross_range_pos:.3e} m, expected < 0")
    elif not np.isclose(total, 0, atol = 1e-9):
        raise ValueError(f"Mirror check FAILED: +5 gave {cross_range_pos:.6e} m, -5 gave {cross_range_neg:.6e} m, sum {total:.3e} m")
    

#-------------- Ballistic Checker for Drag Force -----------------
def ballistic_checker():
    base = Params()
    scaled = replace(base, A = base.A*2, m = base.m*2)
    range1 = solve_summarize(simulate(base))["range"]
    range2 = solve_summarize(simulate(scaled))["range"]
    if not np.isclose(range1, range2, rtol = 1e-6):
        raise RuntimeError("The scaling between mass and area is incorrect")

#------------ Angle Optimization-----------------
def range_at_cond(angle_deg, base: Params):
    return solve_summarize(simulate(replace(base, angle_deg = angle_deg)))["range"]

def angle_sweep(base: Params, angles):
    return np.array([range_at_cond(a, base) for a in angles])

def find_optimal_angle(base: Params):
    low, high = (10, 80)
    margin = 0.5
    res = minimize_scalar(lambda a: -range_at_cond(a, base),
                          bounds=(low, high), method="bounded")
    if res.x < low + margin or res.x > high - margin:
        raise ValueError(f"Best angle {res.x:.2f} deg is at the edge of ({low}, {high})")
    return res.x, -res.fun
#----------- Plotting ------------------
def plot(traj_angles=(20, 30, 45, 60, 75)):
    check_against_vacuum()
    ballistic_checker()
    check_planar_symmetry()
    check_crosswind_sign()


    base = Params()
    angles = np.linspace(5, 85, 100)
    ranges_drag = angle_sweep(base, angles)
    ranges_vac = base.v0**2 * np.sin(2 * np.radians(angles)) / base.g
    best_angle, best_range = find_optimal_angle(base)
    print(f"Optimal angle with drag: {best_angle:.1f} deg, range {best_range:.1f} m")
    print(f"Vacuum optimum: 45.0 deg, range {vacuum_range(base):.1f} m")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8, 8))

    #Top: for trajectories comparison b/w drag(solid) vs vacuum(dashed)
    for a in traj_angles:
        p = replace(base, angle_deg = a )
        sol = simulate(p)
        t = np.linspace(0, sol.t[-1], 300)
        x, y = sol.sol(t)[:2]
        th = np.radians(a)
        tv = np.linspace(0, 2 * p.v0 * np.sin(th) / p.g, 300)
        line, = ax1.plot(x, y, label=f"{a}°")
        ax1.plot(p.v0 * np.cos(th) * tv,
                    p.v0 * np.sin(th) * tv - 0.5 * p.g * tv**2,
                    "--", color= line.get_color(), alpha=0.5)
    ax1.set(xlabel="x (m)", ylabel="y (m)", title="Trajectories (solid: drag, dashed: vacuum)")
    ax1.set_aspect("equal")
    ax1.legend(title="Launch angle")
    ax1.grid(alpha=0.3)
        
    #Bottom: range vs angle
    ax2.plot(angles, ranges_vac, "--", label="Vacuum")
    ax2.plot(angles, ranges_drag, label="With drag")
    ax2.axvline(best_angle, color="gray", ls=":")
    ax2.annotate(f"optimum {best_angle:.1f}°", (best_angle, best_range),
                         textcoords="offset points", xytext=(10, -5))
    ax2.set(xlabel="Launch angle (deg)", ylabel="Range (m)", title="Range vs. launch angle")
    ax2.legend()
    ax2.grid(alpha=0.3)
        
    fig.tight_layout()
    fig.savefig("projectile_sweep.png", dpi=150)
    plt.show()
        
        
if __name__ == "__main__":
    check_against_vacuum()
    print("vacuum check passed")
    check_planar_symmetry()
    print("planar symmetry check passed")
    check_crosswind_sign()
    print("crosswind sign check passed")
