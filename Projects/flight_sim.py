from dataclasses import dataclass, replace
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# ---------- Launch Conditions --------------- 
@dataclass(frozen = True)
class Params:
    v0: float = 100.0       # launch speed, m/s
    angle_deg: float = 45.0  # launch angle above horizontal, degrees
    Cd: float = 0.3         # drag coefficient
    A: float = 0.05         # cross-sectional area, m^2
    m: float = 4.0          # mass, kg
    wind: float = 0.0       # horizontal wind, m/s (positive = tailwind)
    g: float = 9.81         #gravity, m/s^2


#------------ ODE and Function------------------
def simulate(p: Params):
    theta = np.radians(p.angle_deg)
    initial_cond = [0.0, 0.0, p.v0 * np.cos(theta), p.v0 * np.sin(theta)]
    def flight_dynamics(t, cond):
        x, y, vx, vy = cond
        rho = 1.225 * np.exp(-y/8500)
        vx_rel = vx - p.wind
        v_rel = np.sqrt(vx_rel**2 + vy**2)
        F_drag = 0.5 * rho * v_rel**2 * p.Cd * p.A

        if v_rel > 0:
            ax = -(F_drag/p.m) * (vx_rel/v_rel)
            ay = -p.g - (F_drag/p.m) * (vy/v_rel) 
        else:
            ax = 0
            ay = -p.g

        return [vx, vy, ax, ay]
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
    return{"range": sol.y[0, -1], "max_height": sol.y[1].max(), "time": sol.t[-1]}


# ------------ Vacuum Comparison ----------------
def vacuum_range(p: Params):
    return p.v0**2 * np.sin(2 * np.radians(p.angle_deg)) / p.g

def check_against_vacuum():
    p = Params(Cd=0.0)
    numeric = solve_summarize(simulate(p))["range"]
    analytic = vacuum_range(p)
    if not np.isclose(numeric, analytic, rtol=1e-4):
        rel_err = np.abs(numeric - analytic) / np.abs(analytic)
        raise RuntimeError(
            f"Vacuum check FAILED: numeric {numeric:.6f} m vs analytic "
            f"{analytic:.6f} m (relative error {rel_err:.2e})")

#------------ Angle Optimization-----------------
def range_at_cond(angle_deg, base: Params):
    return solve_summarize(simulate(replace(base, angle_deg = angle_deg)))["range"]

def angle_sweep(base: Params, angles):
    return np.array([range_at_cond(a, base) for a in angles])

def find_optimal_angle(base: Params):
    res = minimize_scalar(lambda a: -range_at_cond(a, base),
                          bounds=(10, 80), method="bounded")
    return res.x, -res.fun

#----------- Plotting ------------------
def plot(traj_angles=(20, 30, 45, 60, 75)):
    check_against_vacuum()

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
    plot([10, 25, 50, 70])  

    