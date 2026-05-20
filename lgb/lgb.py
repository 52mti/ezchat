from pathlib import Path
from typing import Optional, Union, List
import socket
import concurrent.futures
from io import BytesIO

import requests
from PIL import Image


def _get_local_ip() -> str:
    """获取本机 IP 地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"


def _parse_ip_prefix(ip_prefix: Optional[str]) -> str:
    """
    解析 IP 前缀
    
    参数:
        ip_prefix: 可以是 None、完整 IP 或网段前缀
            - None: 自动获取本机网段
            - "192.168.3.0": 提取前缀 "192.168.3"
            - "192.168.3": 直接使用
    
    返回: IP 前缀，如 "192.168.3"
    """
    if ip_prefix is None:
        local_ip = _get_local_ip()
        parts = local_ip.split(".")
        return ".".join(parts[:3])
    
    parts = ip_prefix.split(".")
    if len(parts) == 4 and parts[3] == "0":
        return ".".join(parts[:3])
    elif len(parts) == 3:
        return ip_prefix
    else:
        raise ValueError(f"无效的 IP 前缀格式: {ip_prefix}")


def scan_devices(
    ip_prefix: Optional[str] = None,
    port: int = 8888,
    timeout: float = 0.5,
    max_workers: int = 50
) -> List[str]:
    """
    扫描局域网中 8888 端口的设备
    
    参数:
        ip_prefix: IP 前缀，支持以下格式:
            - None: 自动扫描本机所在网段
            - "192.168.3.0": 指定网段
            - "192.168.3": 指定前缀
        port: 端口号，默认 8888
        timeout: 超时时间（秒）
        max_workers: 最大并发数
    
    返回: List[str] 可用设备的 IP 列表
    
    示例:
        devices = scan_devices()              # 扫描本机网段
        devices = scan_devices("192.168.3.0") # 扫描指定网段
        devices = scan_devices("192.168.3")   # 扫描指定前缀
    """
    ip_prefix = _parse_ip_prefix(ip_prefix)
    found_devices = []
    
    def check_ip(ip: str) -> Optional[str]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            if result == 0:
                try:
                    resp = requests.get(f"http://{ip}:{port}/api/status", timeout=timeout)
                    if resp.status_code == 200:
                        return ip
                except:
                    pass
        except:
            pass
        return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_ip, f"{ip_prefix}.{i}"): i
            for i in range(1, 255)
        }
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                found_devices.append(result)
    
    return sorted(found_devices)


class RobotInterface:
    """
    机器人视觉控制系统接口
    
    使用示例:
        robot = RobotInterface("192.168.3.200")
        
        # 获取状态
        status = robot.get_status()
        
        # 获取透视变换图像
        robot.get_transform_image(pre_move=True, save_path="scan.jpg")
        
        # 发送控制指令
        robot.send_command("move,1000,1000", wait=True)
        
        # TTS 语音
        robot.play_audio("你好世界")
        
        # 复位
        robot.reset_device()
        
        # 重启
        robot.restart_system()
    """
    
    def __init__(self, host: str, port: int = 8888, rotate: int = 0):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.rotate = rotate
    
    def get_status(self) -> dict:
        """
        获取设备状态
        
        GET /api/status
        返回: dict 状态信息（包含 IP、MAC、设备ID等）
        """
        resp = requests.get(f"{self.base_url}/api/status")
        resp.raise_for_status()
        return resp.json()
    
    def get_capture_image(
        self,
        pre_move: bool = False,
        save_path: Optional[Union[str, Path]] = None
    ) -> bytes:
        """
        获取原始图像
        
        GET /capture?pre_move=true
        
        参数:
            pre_move: 是否在拍照前移动机械臂到安全位置
            save_path: 保存路径（可选）
        
        返回: bytes JPEG 图像数据
        """
        url = f"{self.base_url}/capture"
        if pre_move:
            url += "?pre_move=true"
        
        resp = requests.get(url)
        resp.raise_for_status()
        
        image_data = resp.content
        if save_path:
            Path(save_path).write_bytes(image_data)
        
        return image_data
    
    def get_transform_image(
        self,
        pre_move: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        rotate: Optional[int] = None
    ) -> bytes:
        """
        获取透视变换图像
        
        GET /transform?pre_move=true
        
        参数:
            pre_move: 是否在拍照前移动机械臂到安全位置，默认 True
            save_path: 保存路径（可选），默认使用 IP 命名，如 "192.168.3.200.jpg"
            rotate: 旋转角度，支持 0/90/180/270，默认使用初始化时设置的值
        
        返回: bytes JPEG 图像数据
        
        示例:
            robot = RobotInterface("192.168.3.200", rotate=90)
            robot.get_transform_image()  # 默认 pre_move=True
            robot.get_transform_image(save_path="scan.jpg", rotate=180)
        """
        url = f"{self.base_url}/transform"
        if pre_move:
            url += "?pre_move=true"
        
        resp = requests.get(url)
        resp.raise_for_status()
        
        image_data = resp.content
        
        rotate_angle = rotate if rotate is not None else self.rotate
        if rotate_angle in [90, 180, 270]:
            img = Image.open(BytesIO(image_data))
            img = img.rotate(-rotate_angle, expand=True)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
        
        if save_path is None:
            save_path = f"{self.host}.jpg"
        
        Path(save_path).write_bytes(image_data)
        
        return image_data
    
    def send_command(
        self,
        cmd: str,
        wait: Union[bool, str, int] = True
    ) -> dict:
        """
        发送控制指令（原始接口）
        
        POST /api/command?wait=true
        Body: cmd=xxx
        
        参数:
            cmd: 指令字符串
            wait: true/false/超时毫秒数
        
        返回: dict 执行结果
        """
        url = f"{self.base_url}/api/command?wait={str(wait).lower() if isinstance(wait, bool) else wait}"
        
        resp = requests.post(url, data={"cmd": cmd})
        resp.raise_for_status()
        return resp.json()
    
    def move_click(
        self,
        x: float,
        y: float,
        click_count: int = 1,
        hold_time: int = 200,
        wait: Union[bool, str, int] = True
    ) -> dict:
        """
        移动并点击
        
        POST /api/command?wait=true
        Body: cmd=move_click,x,y,click_count,hold_time
        
        参数:
            x: X 坐标百分比 (0.0-1.0)
            y: Y 坐标百分比 (0.0-1.0)
            click_count: 点击次数，默认 1
            hold_time: 长按时间（毫秒），默认 200
            wait: true/false/超时毫秒数
        
        返回: dict 执行结果
        
        示例:
            robot.move_click(0.5, 0.5)              # 单击屏幕中心
            robot.move_click(0.5, 0.5, 2, 200)      # 双击
            robot.move_click(0.5, 0.5, 1, 1000)     # 长按 1 秒
        """
        cmd = f"move_click,{x},{y},{click_count},{hold_time}"
        return self.send_command(cmd, wait)
    
    def swipe(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        wait: Union[bool, str, int] = True
    ) -> dict:
        """
        滑屏操作
        
        POST /api/command?wait=true
        Body: cmd=drag,start_x,start_y,end_x,end_y
        
        参数:
            start_x: 起始 X 坐标百分比 (0.0-1.0)
            start_y: 起始 Y 坐标百分比 (0.0-1.0)
            end_x: 结束 X 坐标百分比 (0.0-1.0)
            end_y: 结束 Y 坐标百分比 (0.0-1.0)
            wait: true/false/超时毫秒数
        
        返回: dict 执行结果
        
        示例:
            robot.swipe(0.5, 0.8, 0.5, 0.2)  # 从下往上滑
            robot.swipe(0.2, 0.5, 0.8, 0.5)  # 从左往右滑
        """
        cmd = f"drag,{start_x},{start_y},{end_x},{end_y}"
        return self.send_command(cmd, wait)
    
    def play_audio(
        self,
        text: str,
        wait: bool = True
    ) -> dict:
        """
        播放 TTS 语音（最大 20 个汉字）
        
        POST /api/audio?wait=true
        Body: text=xxx
        
        参数:
            text: 要播放的文本
            wait: 是否等待播放完成
        
        返回: dict 执行结果
        """
        url = f"{self.base_url}/api/audio?wait={str(wait).lower()}"
        
        resp = requests.post(url, data={"text": text})
        resp.raise_for_status()
        return resp.json()
    
    def reset_device(self, wait: bool = True) -> str:
        """
        复位设备
        
        POST /api/reset_device?wait=true
        
        参数:
            wait: 是否等待复位完成
        
        返回: str 响应消息
        """
        url = f"{self.base_url}/api/reset_device?wait={str(wait).lower()}"
        
        resp = requests.post(url)
        resp.raise_for_status()
        return resp.text
    
    def restart_system(self) -> str:
        """
        重启系统
        
        POST /api/restart_system
        
        返回: str 响应消息
        """
        resp = requests.post(f"{self.base_url}/api/restart_system")
        resp.raise_for_status()
        return resp.text
