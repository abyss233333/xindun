# xindun_1110 双臂抓取与工作站流程

本目录当前主流程使用 `I5Dual`（`robot_sim/robots/i5/i5_dual_427_2.py`）、FoundationPose 输出的 `cf.STL` 世界位姿，以及 AUBO 双臂和左侧 PGIA140 夹爪。机械臂负责抓取培养瓶、移至开瓶器、配合右臂执行移液动作，再放回并回到初始关节角。

开瓶器、移液枪和其他工位设备只在仿真中建模；代码没有控制开瓶器、吸液或排液设备的指令。真机运行前须确认实际设备位置、瓶子和夹爪状态，并保证急停可用。

## 当前入口

| 文件 | 用途 |
|---|---|
| `real_grasp.py` | 真机流程；默认只识别和离线规划，添加 `--execute` 后连接双臂与左夹爪，并在规划完成后自动开始执行。 |
| `preview_live_workflow.py` | 相机识别后的完整仿真动画；可直接现场识别，也可读取已有 pose 文件。 |
| `preview_fixed_pose_20261117_005020_438451.py` | 使用脚本中指定的固定 pose 和双臂初始角生成完整动画；不启动相机或连接真机。右臂路径可从本目录的 `right_arm_offline_20261117_005020_438451.json` 读取。 |
| `workcell_followup.py` | 共享的工位目标、运动规划、碰撞检查、动作同步和真机执行逻辑。 |
| `test_workcell_followup.py` | 工作流离线回归测试。 |

`main_new(1)cc.py`、`cartoonall_new_cc.py`、`real.py` 和 `main_new(1)cc_fk_actions.json` 属于旧流程，不是上述实时识别抓取和完整工作站动作的主入口。

## 运行

在 PyCharm 选择 `wrs` 环境，或在 PowerShell 中进入本目录：

```powershell
cd D:\wrs-fujikoshi-main\wrs-fujikoshi-main\xindun_1110
```

现场识别并播放动画：

```powershell
python preview_live_workflow.py
```

读取已有 FoundationPose 原始 `pose.json`，只规划检查而不打开动画窗口：

```powershell
python preview_live_workflow.py --pose-file "D:\FoundationPose-main\FoundationPose-main\foundationpose_world_result\<结果目录>\pose.json" --check-only
```

删除 `--check-only` 会显示动画；`--speed` 只改变动画播放速度，不改变规划或真机速度。使用 `--pose-file` 时应传 FoundationPose 原始 pose，程序会自行应用当前补偿。

真机离线规划与执行：

```powershell
python real_grasp.py
python real_grasp.py --execute
```

不带 `--execute` 时不连接或驱动真机。带 `--execute` 时，程序在规划成功后会**自动开始运动，不会再等待输入 `EXECUTE`**。运行前须先确认工作区清空、右夹爪已夹好移液枪、目标与相机标定一致且急停可用。右夹爪不由此程序开合。

固定 pose 动画：

```powershell
python preview_fixed_pose_20261117_005020_438451.py
```

该脚本中的 pose 文件位置、初始关节角和开瓶器 X 偏移写在脚本顶部。若右臂离线 JSON 缺失或与当前版本、速度、初始角等条件不匹配，脚本会重新规划并保存；有效时复用缓存。左臂仍按当前 pose 和模型重新规划。

## 位姿与规划设置

- 目标网格为 `427/meshes/cf.STL`；FoundationPose 输出的世界坐标以米为单位。
- `real_grasp.py` 将物体局部坐标补偿 `Y +10 mm、Z +5 mm`，并将候选姿态的物体 Z 轴对齐到参考样本方向，保留绕 Z 轴的朝向。`preview_live_workflow.py` 与真机共用这套 pose 处理。
- 参考姿态文件默认是 `20261113_220757_649151/pose.json`，可通过环境变量 `XINDUN_REFERENCE_ORIENTATION_POSE` 指定其他文件。FoundationPose 脚本、Python 环境、网格和 SAM 权重的位置配置在 `real_grasp.py` 顶部；本机部署路径不同时需在那里更新。
- 工位目标和开瓶器模型位置定义在 `workcell_followup.py`。真机和实时动画将开瓶器目标沿世界 `+X` 偏移 `75 mm`，固定 pose 动画也应用相同偏移。
- 当前 `MOVE_VEL=0.8`、`MOVE_ACC=1.0`。路径通过 `move_jntspace_path` 执行；直线路径以连续 IK 路点表示。

## 碰撞检查范围

- 实时动画和真机使用的工作站规划会把三个 `bottle_openning.stl` 模型加入步骤 5、8、27 的左臂 RRTC 障碍物列表；固定 pose 动画也开启相同的开瓶器障碍物设置。
- RRTC 同时检查机器人自碰撞、持有物和障碍物。右手持有的 `pipette_gun.stl` 通过 `robot.hold()` 注册，因此它也参与这些碰撞检查。
- 桌面 `objects/table.stl` 用作规划碰撞模型；桌面与机器人手臂/夹爪的碰撞按代码策略忽略。瓶子在放回相关步骤 23–26 对桌面障碍有特定接触豁免，其他对象和碰撞规则继续检查。
- 碰撞检查采用选择性策略：检查最初的双臂同步动作及左臂 RRTC 路径；其他固定路径不会借碰撞检测重新搜索替代轨迹。此策略不等同于对真机全流程进行实时防撞。
- 固定 pose 脚本会在缓存右臂路径时重用有效 JSON；缓存校验及其他非 RRTC 固定路径不代表整条真机路径都经过完整碰撞复核。

## 设备与注意事项

- 左臂 AUBO：`192.168.110.12:30004`；右臂 AUBO：`192.168.110.11:30004`。
- 左夹爪默认串口为 `com10`，PGIA140 ID 为 `2`；可用环境变量 `XINDUN_GRIPPER_PORT` 覆盖串口。
- 真机运行需安装项目使用的 AUBO SDK 和 PGIA140 驱动，并确认 FoundationPose 摄像头、Python 环境、SAM 权重及 mesh 路径可用。
- 固定 pose 动画使用预设初始关节角；真机执行使用采集 pose 后读取到的当前关节角。两者初态不同，动画不能代替现场安全确认或真机验证。

测试工作流：

```powershell
python -m unittest xindun_1110.test_workcell_followup
```
