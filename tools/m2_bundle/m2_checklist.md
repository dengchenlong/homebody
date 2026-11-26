# M2 真机冒烟清单（现场打勾用）

对应实施计划 M2 验收；完整环境自检跑 `m2_setup.sh`，全绿才往下走。

## 0. 行前（服务器上已完成）

- [ ] `bash tools/kuavo_robot_pack.sh` 打包，tar 拷到笔记本
- [ ] 笔记本：`python3-venv` + `pip install numpy scipy`；Ubuntu 20.04 装 `ros-noetic-ros-base`
- [ ] kuavo_msgs 来源确定：sshfs 挂机器人 devel 树到 `~/robot_ws_devel`（脚本自动探测）

## 1. 上电与站立

- [ ] 机器人上电，等待控制器起（现场惯例；真机注意 `real:=true` 相关参数）
- [ ] 机器人站稳、MPC 无报错；**周围清场**
- [ ] `rosservice call /enable_control "data: true"` 使能，**手离开机器人**
      （该服务是 `std_srvs/SetBool`：`data: true` 恢复控制，`data: false` 是软暂停；空 `"{}"` 等于 false，会暂停机器人）

## 2. 通道检查（只读，不动机器人）

- [ ] `bash tools/m2_setup.sh` 全绿（重点：`/sensors_data_raw` > 50 Hz、rtt < 1 ms 有线）
- [ ] 三话题在：`/sensors_data_raw`、`/odom`、`/humanoid/mpc/arm_control_mode`
- [ ] Wi-Fi 到达率记录（备用链路）：`rostopic hz /sensors_data_raw` —— **回填 profile**

## 3. 软件验收（机器人不动）

- [ ] `python3 tools/m2_limit_clip_test.py` —— 14 项超限全部拒绝 + epoch 过期拒绝
- [ ] 记录 URDF 限位与实机出厂限位的差异（若有告警）——**回填 profile**

## 4. 运动（按顺序，每步之间人撤到安全位）

- [ ] **③左臂**：一次只动左臂，0.15 rad 预设，ramp 3 s，reach time 等待 + 视觉确认到位
- [ ] **③右臂**：同上
- [ ] **④前进**：0.1 m/s × 1 s，**人跟着走，急停手 ready**（Ctrl+C = 立即零速）
- [ ] 停止后确认零速（`rostopic echo /cmd_vel` 1 s 内归零）
- [ ] rate clamp 观察远目标：手臂缓趋近不跳变

## 5. 急停预案（先读再动手）

1. **Ctrl+C**：demo SIGINT → `stop()` 零速 → deadman 兜底（1 s）
2. 遥控器急停（若现场配置）优先级最高
3. 最后手段：kill 控制器 / 断使能（机器人会蹲/锁，注意环境）

## 6. 回填（回服务器后）

- [ ] 实际关节限位余量、reach time 表现、话题到达率 → `profile.py` + 进度日志
- [ ] 任何与 URDF 不符的现场事实 → `assets/robot/kuavo_5w/`
- [ ] 进度日志追加 M2 记录；issue #1 留言
