#!/usr/bin/env python3
"""
data_recorder.py — Graba posición deseada vs actual y calcula RMSE
Equipo: 4 DE ASADA  |  TE3001B
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros
import numpy as np
import csv, os, math
from datetime import datetime

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')

        self.declare_parameter('experiment', 'baseline')  # baseline|sine|gaussian
        self.declare_parameter('output_dir', '/tmp')
        self.declare_parameter('rate_hz',    50.0)

        self.exp    = self.get_parameter('experiment').value
        self.outdir = self.get_parameter('output_dir').value
        rate        = self.get_parameter('rate_hz').value

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.desired = None
        self.records = []   # lista de [t, xd, yd, zd, xa, ya, za]

        self.sub = self.create_subscription(
            PoseStamped, '/desired_pose', self.desired_cb, 10)
        self.timer = self.create_timer(1.0/rate, self.record)

        self.t0 = None
        self.get_logger().info(f'DataRecorder: experimento={self.exp}')

    def desired_cb(self, msg):
        self.desired = msg

    def record(self):
        if self.desired is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                'link_base', 'link_eef', rclpy.time.Time())
        except Exception:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0

        xd = self.desired.pose.position.x
        yd = self.desired.pose.position.y
        zd = self.desired.pose.position.z
        xa = tf.transform.translation.x
        ya = tf.transform.translation.y
        za = tf.transform.translation.z

        self.records.append([t, xd, yd, zd, xa, ya, za])

    def save_and_evaluate(self):
        if not self.records:
            self.get_logger().warn('No hay datos grabados.')
            return

        data = np.array(self.records)
        t   = data[:,0]
        xd, yd, zd = data[:,1], data[:,2], data[:,3]
        xa, ya, za = data[:,4], data[:,5], data[:,6]

        ex = xd - xa
        ey = yd - ya
        ez = zd - za

        rmse_x = math.sqrt(np.mean(ex**2))
        rmse_y = math.sqrt(np.mean(ey**2))
        rmse_z = math.sqrt(np.mean(ez**2))
        rmse_total = math.sqrt(np.mean(ex**2 + ey**2 + ez**2))
        max_err = np.max(np.sqrt(ex**2 + ey**2 + ez**2))

        print(f'\n===== RESULTADOS: {self.exp} =====')
        print(f'  RMSE X:     {rmse_x*1000:.3f} mm')
        print(f'  RMSE Y:     {rmse_y*1000:.3f} mm')
        print(f'  RMSE Z:     {rmse_z*1000:.3f} mm')
        print(f'  RMSE Total: {rmse_total*1000:.3f} mm')
        print(f'  Error Max:  {max_err*1000:.3f} mm')

        # Guardar CSV
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(self.outdir, f'{self.exp}_{ts}.csv')
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t','xd','yd','zd','xa','ya','za'])
            w.writerows(self.records)
        print(f'  Datos guardados: {fname}')
        return fname

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.save_and_evaluate()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
