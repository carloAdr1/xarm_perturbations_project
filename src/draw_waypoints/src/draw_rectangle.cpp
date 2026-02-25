/**
 * draw_rectangle.cpp  —  Equipo: 4 DE ASADA
 * TE3001B Reto Semana 2
 *
 * PLANO ARBITRARIO: captura 3 puntos para definir cualquier plano
 * (horizontal, frontal, lateral, inclinado).
 *
 * Matemáticas:
 *   P1 = origen del plano
 *   u  = normalize(P2 - P1)  → eje "derecha" del plano
 *   v  = normalize(P3 - P1)  → eje "arriba"  del plano
 *   n  = u × v               → normal (hacia donde apunta la pluma)
 *
 * Para dibujar en el plano:
 *   punto_mundo = P1 + s*u + t*v
 *   donde s ∈ [0, ancho], t ∈ [0, alto]
 *
 * La orientación del efector se calcula para que apunte
 * en dirección -n (perpendicular al plano, hacia adentro).
 */

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <array>

using namespace std::chrono_literals;
using WpList = std::vector<geometry_msgs::msg::Pose>;
using MG     = moveit::planning_interface::MoveGroupInterface;
using Vec3   = std::array<double, 3>;

// ==========================================================
// Álgebra de vectores 3D
// ==========================================================
Vec3 vsub(Vec3 a, Vec3 b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
Vec3 vadd(Vec3 a, Vec3 b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
Vec3 vscale(Vec3 a, double s) { return {a[0]*s, a[1]*s, a[2]*s}; }
double vdot(Vec3 a, Vec3 b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
double vnorm(Vec3 a) { return std::sqrt(vdot(a,a)); }
Vec3 vnormalize(Vec3 a) { double n=vnorm(a); return {a[0]/n,a[1]/n,a[2]/n}; }
Vec3 vcross(Vec3 a, Vec3 b) {
    return { a[1]*b[2]-a[2]*b[1],
             a[2]*b[0]-a[0]*b[2],
             a[0]*b[1]-a[1]*b[0] };
}

// Convierte matrix de rotación (u,v,n como columnas) a cuaternión
// R = [u | v | n]  donde n es la normal del plano
struct Quat { double x,y,z,w; };
Quat rot_to_quat(Vec3 u, Vec3 v, Vec3 n)
{
    // Matriz de rotación 3x3 → cuaternión
    double m00=u[0], m10=u[1], m20=u[2];
    double m01=v[0], m11=v[1], m21=v[2];
    double m02=n[0], m12=n[1], m22=n[2];

    double trace = m00+m11+m22;
    Quat q;
    if (trace > 0) {
        double s = 0.5/std::sqrt(trace+1.0);
        q.w = 0.25/s;
        q.x = (m21-m12)*s;
        q.y = (m02-m20)*s;
        q.z = (m10-m01)*s;
    } else if (m00>m11 && m00>m22) {
        double s = 2.0*std::sqrt(1.0+m00-m11-m22);
        q.w = (m21-m12)/s;
        q.x = 0.25*s;
        q.y = (m01+m10)/s;
        q.z = (m02+m20)/s;
    } else if (m11>m22) {
        double s = 2.0*std::sqrt(1.0+m11-m00-m22);
        q.w = (m02-m20)/s;
        q.x = (m01+m10)/s;
        q.y = 0.25*s;
        q.z = (m12+m21)/s;
    } else {
        double s = 2.0*std::sqrt(1.0+m22-m00-m11);
        q.w = (m10-m01)/s;
        q.x = (m02+m20)/s;
        q.y = (m12+m21)/s;
        q.z = 0.25*s;
    }
    return q;
}

// ==========================================================
// Variables globales del plano
// ==========================================================
static Vec3   P_ORIGIN;      // origen del plano (P1)
static Vec3   U_AXIS;        // eje derecha (normalizado)
static Vec3   V_AXIS;        // eje arriba  (normalizado)
static Vec3   N_AXIS;        // normal del plano
static double PLANE_W;       // ancho del plano en metros
static double PLANE_H;       // alto  del plano en metros
static double LIFT_DIST;     // distancia de levantamiento (m)
static Quat   EEF_QUAT;      // orientación fija del efector

// ==========================================================
// Convierte coordenadas locales [0..1] → pose en el mundo
// s: posición horizontal [0=izq, 1=der]
// t: posición vertical   [0=abajo, 1=arriba]
// lift: si true, desplaza en dirección normal (pluma arriba)
// ==========================================================
geometry_msgs::msg::Pose plane_pose(double s, double t, bool lift = false)
{
    // Punto en el plano
    Vec3 pt = vadd(P_ORIGIN,
                   vadd(vscale(U_AXIS, s * PLANE_W),
                        vscale(V_AXIS, t * PLANE_H)));
    // Si levantamos, alejamos en dirección normal
    if (lift) pt = vadd(pt, vscale(N_AXIS, -LIFT_DIST));

    geometry_msgs::msg::Pose p;
    p.position.x    = pt[0];
    p.position.y    = pt[1];
    p.position.z    = pt[2];
    p.orientation.x = EEF_QUAT.x;
    p.orientation.y = EEF_QUAT.y;
    p.orientation.z = EEF_QUAT.z;
    p.orientation.w = EEF_QUAT.w;
    return p;
}

// Pose dentro de una columna de letra
// col_l, col_r: fracción [0..1] del ancho asignada a la letra
// cx: posición en la columna [0=izq, 1=der]
// cy: altura [0=abajo, 1=arriba]
geometry_msgs::msg::Pose LP(double col_l, double col_r,
                             double cx, double cy, bool lift=false)
{
    double s = col_l + cx * (col_r - col_l);
    return plane_pose(s, cy, lift);
}

// ==========================================================
// draw_stroke: Cartesian path con fallback pose-goal
// ==========================================================
void draw_stroke(MG & mg, const WpList & wps, const rclcpp::Logger & log)
{
    if (wps.empty()) return;
    moveit_msgs::msg::RobotTrajectory traj;
    double frac = mg.computeCartesianPath(wps, 0.002, 0.0, traj);
    RCLCPP_INFO(log, "  fraction=%.2f  pts=%zu", frac, wps.size());
    if (frac > 0.80) {
        MG::Plan plan;
        plan.trajectory_ = traj;
        mg.execute(plan);
    } else {
        RCLCPP_WARN(log, "  fraction baja → pose-goals individuales");
        for (auto & wp : wps) {
            mg.setPoseTarget(wp);
            MG::Plan plan;
            if (mg.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
                mg.execute(plan);
        }
    }
}

// ==========================================================
// pen_up_goto: levanta (en dirección normal), viaja, baja
// ==========================================================
void pen_up_goto(MG & mg, double s, double t, const rclcpp::Logger & log)
{
    // 1) Levantar desde posición actual en dirección normal
    auto cur = mg.getCurrentPose().pose;
    Vec3 cur_pt = {cur.position.x, cur.position.y, cur.position.z};
    Vec3 lifted  = vadd(cur_pt, vscale(N_AXIS, -LIFT_DIST));

    geometry_msgs::msg::Pose lift_pose;
    lift_pose.position.x    = lifted[0];
    lift_pose.position.y    = lifted[1];
    lift_pose.position.z    = lifted[2];
    lift_pose.orientation   = cur.orientation;
    draw_stroke(mg, {lift_pose}, log);

    // 2) Viajar al destino (levantado)
    mg.setPoseTarget(plane_pose(s, t, true));
    MG::Plan plan;
    if (mg.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
        mg.execute(plan);

    // 3) Bajar al plano
    draw_stroke(mg, {plane_pose(s, t, false)}, log);
}

// Versión con columna de letra
void pen_up_goto_col(MG & mg, double col_l, double col_r,
                     double cx, double cy, const rclcpp::Logger & log)
{
    double s = col_l + cx * (col_r - col_l);
    pen_up_goto(mg, s, cy, log);
}

// ==========================================================
// LETRAS
// ==========================================================
void draw_4(MG & mg, double l, double r, const rclcpp::Logger & log)
{
    pen_up_goto_col(mg, l, r, 0.8, 0.9, log);
    draw_stroke(mg, { LP(l,r,0.8,0.9), LP(l,r,0.8,0.1) }, log);

    pen_up_goto_col(mg, l, r, 0.0, 0.5, log);
    draw_stroke(mg, { LP(l,r,0.0,0.5), LP(l,r,0.8,0.5) }, log);

    pen_up_goto_col(mg, l, r, 0.0, 0.9, log);
    draw_stroke(mg, { LP(l,r,0.0,0.9), LP(l,r,0.0,0.5) }, log);
}

void draw_A(MG & mg, double l, double r, const rclcpp::Logger & log)
{
    pen_up_goto_col(mg, l, r, 0.0, 0.1, log);
    draw_stroke(mg, {
        LP(l,r,0.0,0.1), LP(l,r,0.5,0.9), LP(l,r,1.0,0.1)
    }, log);

    pen_up_goto_col(mg, l, r, 0.2, 0.5, log);
    draw_stroke(mg, { LP(l,r,0.2,0.5), LP(l,r,0.8,0.5) }, log);
}

void draw_S(MG & mg, double l, double r, const rclcpp::Logger & log)
{
    pen_up_goto_col(mg, l, r, 1.0, 0.9, log);
    draw_stroke(mg, {
        LP(l,r,1.0,0.9), LP(l,r,0.0,0.9),
        LP(l,r,0.0,0.5), LP(l,r,1.0,0.5),
        LP(l,r,1.0,0.1), LP(l,r,0.0,0.1)
    }, log);
}

void draw_D(MG & mg, double l, double r, const rclcpp::Logger & log)
{
    pen_up_goto_col(mg, l, r, 0.0, 0.1, log);
    draw_stroke(mg, {
        LP(l,r,0.0,0.1), LP(l,r,0.0,0.9),
        LP(l,r,0.5,0.9), LP(l,r,1.0,0.7),
        LP(l,r,1.0,0.3), LP(l,r,0.5,0.1),
        LP(l,r,0.0,0.1)
    }, log);
}

// ==========================================================
// main
// ==========================================================
int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions opts;
    opts.automatically_declare_parameters_from_overrides(true);
    auto node = std::make_shared<rclcpp::Node>("draw_rectangle", opts);
    auto log  = node->get_logger();

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    std::thread spin_t([&exec](){ exec.spin(); });

    // ── Esperar joint_states ──────────────────────────────
    RCLCPP_INFO(log, "Esperando joint_states...");
    bool got = false;
    auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10,
        [&got](sensor_msgs::msg::JointState::SharedPtr m){
            if (m->header.stamp.sec > 0) got = true;
        });
    auto t0 = node->now();
    while (!got && (node->now()-t0).seconds() < 15.0)
        std::this_thread::sleep_for(100ms);
    if (!got) {
        RCLCPP_ERROR(log, "No llegaron joint_states");
        exec.cancel(); spin_t.join(); rclcpp::shutdown(); return 1;
    }

    // ── MoveGroup ─────────────────────────────────────────
    auto mg = std::make_shared<MG>(node, "lite6");
    mg->setMaxVelocityScalingFactor(0.15);
    mg->setMaxAccelerationScalingFactor(0.15);
    mg->setPlanningTime(10.0);
    mg->setNumPlanningAttempts(5);

    // ── TEACH MODE: 3 puntos definen el plano ─────────────
    RCLCPP_INFO(log, "=== TEACH MODE — Plano Arbitrario ===");
    RCLCPP_INFO(log, "3 puntos definen cualquier plano:");
    RCLCPP_INFO(log, "  P1 = esquina inf-izq  (origen)");
    RCLCPP_INFO(log, "  P2 = esquina inf-der  (define eje derecha)");
    RCLCPP_INFO(log, "  P3 = esquina sup-izq  (define eje arriba)");
    RCLCPP_INFO(log, "Funciona para plano horizontal, frontal, lateral o inclinado.");

    std::vector<geometry_msgs::msg::Pose> pts(3);
    std::vector<std::string> names = {
        "P1 — inf-izq (ORIGEN)",
        "P2 — inf-der (eje derecha)",
        "P3 — sup-izq (eje arriba)"
    };

    for (int i = 0; i < 3; i++) {
        std::cout << "\n>>> Mueve el robot a " << names[i]
                  << " y presiona ENTER: " << std::flush;
        std::cin.get();
        pts[i] = mg->getCurrentPose().pose;
        RCLCPP_INFO(log, "  %s capturado: x=%.4f  y=%.4f  z=%.4f",
            names[i].c_str(),
            pts[i].position.x,
            pts[i].position.y,
            pts[i].position.z);
    }

    // ── Calcular ejes del plano ───────────────────────────
    P_ORIGIN = { pts[0].position.x, pts[0].position.y, pts[0].position.z };
    Vec3 p2  = { pts[1].position.x, pts[1].position.y, pts[1].position.z };
    Vec3 p3  = { pts[2].position.x, pts[2].position.y, pts[2].position.z };

    Vec3 raw_u = vsub(p2, P_ORIGIN);   // P1→P2 = eje derecha
    Vec3 raw_v = vsub(p3, P_ORIGIN);   // P1→P3 = eje arriba

    PLANE_W = vnorm(raw_u);   // ancho real del plano
    PLANE_H = vnorm(raw_v);   // alto  real del plano

    U_AXIS = vnormalize(raw_u);
    V_AXIS = vnormalize(raw_v);
    N_AXIS = vnormalize(vcross(U_AXIS, V_AXIS));  // normal del plano

    LIFT_DIST = 0.03;   // 3 cm fuera del plano entre trazos

    // Calcular orientación del efector para apuntar en -N
    // El efector apunta en su eje Z local. Queremos Z_local = -N.
    // Usamos u como X_local, v como Y_local, -n como Z_local.
    Vec3 neg_n = vscale(N_AXIS, -1.0);
    EEF_QUAT = rot_to_quat(U_AXIS, V_AXIS, neg_n);

    RCLCPP_INFO(log, "Plano calculado:");
    RCLCPP_INFO(log, "  Origen:  (%.4f, %.4f, %.4f)", P_ORIGIN[0], P_ORIGIN[1], P_ORIGIN[2]);
    RCLCPP_INFO(log, "  U (der): (%.4f, %.4f, %.4f)", U_AXIS[0], U_AXIS[1], U_AXIS[2]);
    RCLCPP_INFO(log, "  V (arr): (%.4f, %.4f, %.4f)", V_AXIS[0], V_AXIS[1], V_AXIS[2]);
    RCLCPP_INFO(log, "  Normal:  (%.4f, %.4f, %.4f)", N_AXIS[0], N_AXIS[1], N_AXIS[2]);
    RCLCPP_INFO(log, "  Ancho=%.4f m   Alto=%.4f m", PLANE_W, PLANE_H);
    RCLCPP_INFO(log, "  EEF quat: x=%.4f y=%.4f z=%.4f w=%.4f",
                EEF_QUAT.x, EEF_QUAT.y, EEF_QUAT.z, EEF_QUAT.w);

    // Detectar tipo de plano automáticamente
    double nx = std::abs(N_AXIS[0]);
    double ny = std::abs(N_AXIS[1]);
    double nz = std::abs(N_AXIS[2]);
    if (nz > 0.8)
        RCLCPP_INFO(log, "  Tipo: HORIZONTAL (como mesa)");
    else if (ny > 0.8)
        RCLCPP_INFO(log, "  Tipo: FRONTAL (como pizarrón)");
    else if (nx > 0.8)
        RCLCPP_INFO(log, "  Tipo: LATERAL (como pared)");
    else
        RCLCPP_INFO(log, "  Tipo: INCLINADO");

    std::cout << "\n>>> Presiona ENTER para iniciar el dibujo: " << std::flush;
    std::cin.get();

    // ── DIBUJO: 4 A S A D A ───────────────────────────────
    RCLCPP_INFO(log, "=== Dibujando: 4ASADA en plano arbitrario ===");

    RCLCPP_INFO(log, "--- 4 ---");
    draw_4(*mg, 0.00, 0.14, log);

    RCLCPP_INFO(log, "--- A ---");
    draw_A(*mg, 0.17, 0.31, log);

    RCLCPP_INFO(log, "--- S ---");
    draw_S(*mg, 0.34, 0.48, log);

    RCLCPP_INFO(log, "--- A ---");
    draw_A(*mg, 0.51, 0.65, log);

    RCLCPP_INFO(log, "--- D ---");
    draw_D(*mg, 0.68, 0.82, log);

    RCLCPP_INFO(log, "--- A ---");
    draw_A(*mg, 0.85, 0.99, log);

    // Levantar al terminar
    {
        auto cur = mg->getCurrentPose().pose;
        Vec3 cur_pt = {cur.position.x, cur.position.y, cur.position.z};
        Vec3 lifted  = vadd(cur_pt, vscale(N_AXIS, -LIFT_DIST));
        geometry_msgs::msg::Pose lp;
        lp.position.x  = lifted[0];
        lp.position.y  = lifted[1];
        lp.position.z  = lifted[2];
        lp.orientation = cur.orientation;
        draw_stroke(*mg, {lp}, log);
    }

    RCLCPP_INFO(log, "¡Equipo 4 DE ASADA completó el reto!");
    exec.cancel(); spin_t.join(); rclcpp::shutdown();
    return 0;
}
