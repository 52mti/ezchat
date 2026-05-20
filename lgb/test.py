from lgb import RobotInterface, scan_devices


def test_scan():
    print("=== 扫描设备 ===")
    
    # 不传参数，自动扫描本机网段
    devices = scan_devices()
    print(f"本机网段设备: {devices}")
    
    # 指定网段
    # devices = scan_devices("192.168.3.0")
    # print(f"指定网段设备: {devices}")
    
    print()
    return devices


def test():
    # 初始化时设置旋转角度
    robot = RobotInterface("192.168.3.53", rotate=90)
    
    # 1. 获取设备状态
    print("=== 获取设备状态 ===")
    status = robot.get_status()
    print(f"状态: {status}")
    print()
    
    # 2. 获取透视变换图像（默认 pre_move=True，用 IP 命名）
    print("=== 获取透视变换图像 ===")
    robot.get_transform_image()
    print("图像已保存到 192.168.3.200.jpg")
    print()
    
    # 3. 移动并点击
    print("=== 移动并点击 ===")
    # 单击（使用默认值：click_count=1, hold_time=200）
    result = robot.move_click(0.5, 0.5)
    print(f"单击结果: {result}")
    
    # 双击
    result = robot.move_click(0.5, 0.5, 2, 200)
    print(f"双击结果: {result}")
    
    # 长按 1 秒
    result = robot.move_click(0.5, 0.5, 1, 1000)
    print(f"长按结果: {result}")
    print()
    
    # 4. 滑屏操作
    print("=== 滑屏操作 ===")
    result = robot.swipe(0.5, 0.8, 0.5, 0.2)
    print(f"结果: {result}")
    print()
    
    # 5. TTS 语音
    print("=== TTS 语音 ===")
    result = robot.play_audio("你好世界")
    print(f"结果: {result}")
    print()
    
    # 6. 复位设备
    print("=== 复位设备 ===")
    result = robot.reset_device()
    print(f"结果: {result}")
    print()


if __name__ == "__main__":
    test_scan()
    test()
