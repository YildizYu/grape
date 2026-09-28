#!/usr/bin/env python3
"""供采摘流程工作线程使用的剪刀服务客户端，也可单独命令行调用。

ROS executor 必须在另一个线程持续 spin。不要在 ROS 回调中阻塞调用。
只有调用 open()/close() 才会发送指令；创建对象不会自动操作剪刀。
不订阅 /target_pose，不根据固定延时判断机械臂是否到位。
"""

import argparse
import math
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from grasp_harvest_control.srv import SetScissorState


def _positive_timeout(value):
    """在创建 ROS 节点之前检查命令行等待时间。"""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('等待时间必须为有限正数') from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('等待时间必须为有限正数')
    return value


class ScissorError(RuntimeError):
    """服务不可用、被禁用、忙碌或设备动作失败。"""


class ScissorStateUnknown(ScissorError):
    """命令可能已经执行，禁止自动重试或立刻发送相反命令。"""


class ScissorServiceClient:
    def __init__(self, node, timeout_sec=5.0, enabled=True,
                 service_name='/scissor/set_state'):
        timeout_sec = float(timeout_sec)
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError('timeout_sec 必须为有限正数')
        self.node = node
        self.timeout_sec = timeout_sec
        self.enabled = enabled
        self.state_unknown = False
        self._lock = threading.Lock()
        # 单独的回调组，避免与其他节点回调共享互斥组。
        self._group = MutuallyExclusiveCallbackGroup()
        self._client = node.create_client(
            SetScissorState, service_name, callback_group=self._group
        ) if enabled else None

    def is_ready(self):
        """仅查询服务是否可用，不操作剪刀，也不读取物理开合状态。"""
        return (self.enabled and not self.state_unknown
                and self._client.service_is_ready())

    def wait_ready(self, timeout_sec=3.0):
        """等待服务上线后返回可用状态；不操作剪刀，不读取物理开合状态。"""
        if not self.enabled:
            return False
        self._client.wait_for_service(timeout_sec=timeout_sec)
        return self.is_ready()

    def reset(self):
        """人工确认剪刀实际状态后清除未知锁存，恢复后续动作。

        仅供人工核实设备后调用；自动调用会使 ScissorStateUnknown
        的防误动作保护失效（保安全开剪除外，见 dobot_client._ros_action）。
        """
        self.state_unknown = False

    def open(self, abort_event=None):
        """打开剪刀；成功返回服务响应，失败抛出异常。"""
        return self._execute(SetScissorState.Request.OPEN, '打开', abort_event)

    def close(self, abort_event=None):
        """关闭剪刀；应在本次剪切目标确认为到位后调用。"""
        return self._execute(SetScissorState.Request.CLOSE, '关闭', abort_event)

    def _execute(self, command, action, abort_event):
        if not self.enabled:
            raise ScissorError('detect-only/禁用模式：未发送剪刀指令')
        if not self._lock.acquire(blocking=False):
            raise ScissorError('剪刀客户端忙碌，本次请求未发送')

        future = None
        request_attempted = False
        try:
            if self.state_unknown:
                raise ScissorStateUnknown('前次动作状态未知，请人工核实设备后恢复流程')
            if abort_event is not None and abort_event.is_set():
                raise ScissorError('流程已经中止，本次请求未发送')
            if not rclpy.ok():
                raise ScissorError('ROS 已关闭，本次请求未发送')
            if not self._client.wait_for_service(timeout_sec=1.0):
                raise ScissorError('剪刀服务不可用，本次请求未发送')
            # 等待服务期间也可能发生 ABORT，发送前再次检查。
            if abort_event is not None and abort_event.is_set():
                raise ScissorError('流程已经中止，本次请求未发送')
            if not rclpy.ok():
                raise ScissorError('ROS 在等待服务期间关闭，本次请求未发送')

            request = SetScissorState.Request()
            request.command = command
            completed = threading.Event()
            completed_at = None
            deadline = time.monotonic() + self.timeout_sec

            def on_completed(_):
                nonlocal completed_at
                # 记录客户端处理完成通知的时间，不是设备动作时间。
                completed_at = time.monotonic()
                completed.set()

            try:
                # call_async 被中断时，也可能已经将请求发出。
                request_attempted = True
                future = self._client.call_async(request)
                future.add_done_callback(on_completed)
            except Exception as exc:
                raise ScissorStateUnknown(f'发送服务请求异常：{exc}') from exc

            self.node.get_logger().info(f'已请求{action}剪刀，等待服务执行结果')

            # 阻塞的是流程工作线程；ROS executor 在后台继续接收响应。
            while True:
                if abort_event is not None and abort_event.is_set():
                    raise ScissorStateUnknown('命令发出后流程中止；剪刀可能仍在动作')
                if not rclpy.ok():
                    raise ScissorStateUnknown('ROS 在等待期间关闭，剪刀状态未知')
                if completed.is_set():
                    if completed_at is None or completed_at > deadline:
                        raise ScissorStateUnknown('完成通知超过等待时限，剪刀状态需确认')
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ScissorStateUnknown('服务响应超时；不重试，剪刀可能仍在动作')
                completed.wait(min(0.05, remaining))

            try:
                if future.cancelled():
                    raise RuntimeError('服务请求被本地取消')
                response = future.result()
                if response is None:
                    raise RuntimeError('服务返回空结果')
                if not response.success:
                    raise RuntimeError(
                        f'设备返回失败：code={response.error_code}, {response.message}'
                    )
            except Exception as exc:
                raise ScissorStateUnknown(str(exc)) from exc

            self.node.get_logger().info(f'{action}剪刀成功：{response.message}')
            return response
        except BaseException as exc:
            # KeyboardInterrupt/SystemExit 不属于 Exception，也必须先清理并锁定。
            # 处理完重新抛出，不吞掉用户中断。
            if request_attempted:
                self.state_unknown = True
                self._discard(future)
                if isinstance(exc, Exception) and not isinstance(exc, ScissorError):
                    raise ScissorStateUnknown(f'请求发出后异常：{exc}') from exc
            elif isinstance(exc, Exception) and not isinstance(exc, ScissorError):
                raise ScissorError(f'请求准备失败，未发送指令：{exc}') from exc
            raise
        finally:
            self._lock.release()

    def _discard(self, future):
        if future is None:
            return
        try:
            self._client.remove_pending_request(future)
        except Exception as exc:
            self.node.get_logger().warning(f'移除本地请求失败：{exc}')
        # 移除操作失败时，仍尝试取消本地 Future。
        try:
            future.cancel()
        except Exception as exc:
            self.node.get_logger().warning(f'取消本地 Future 失败：{exc}')
        # 此操作不会停止 C++、RS485 或硬件动作。


def main():
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.node import Node

    parser = argparse.ArgumentParser(description='直接调用原有剪刀服务')
    parser.add_argument('--command', choices=('open', 'close'), required=True)
    # 默认与现场配置 service_timeout_s (25s) 一致: 实测合剪全程 ~8s,
    # 5s 太短会误报"服务响应超时"
    parser.add_argument('--timeout', type=_positive_timeout, default=25.0)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = None
    executor = None
    spin_thread = None
    try:
        node = Node('scissor_command_client')
        scissors = ScissorServiceClient(node, timeout_sec=args.timeout)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def spin():
            try:
                executor.spin()
            except ExternalShutdownException:
                pass

        spin_thread = threading.Thread(target=spin, daemon=True)
        spin_thread.start()
        if args.command == 'open':
            scissors.open()
        else:
            scissors.close()
        return 0
    except ScissorError as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        return 1
    except (KeyboardInterrupt, ExternalShutdownException):
        return 130
    finally:
        if executor is not None:
            executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
