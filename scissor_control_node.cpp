#include <rclcpp/rclcpp.hpp>
#include "grasp_harvest_control/rs485_serial.hpp"
#include "grasp_harvest_control/scissor_protocol.hpp"
#include "grasp_harvest_control/srv/set_scissor_state.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

/*
ros2 service call /scissor/set_state \
grasp_harvest_control/srv/SetScissorState \
"{command: 1}"



0 = OPEN
1 = CLOSE
*/


class ScissorControlNode : public rclcpp::Node
{
public:

    using SetScissorState = grasp_harvest_control::srv::SetScissorState;

    ScissorControlNode() : Node("scissor_control_node")
    {
        port_ = declare_parameter<std::string>("port",  "/dev/ttyXRUSB0");
        baudrate_ = declare_parameter<int>("baudrate",  9600);
        timeout_ms_ =   declare_parameter<int>("timeout_ms",    10000);
        hw_flow_control_ =  declare_parameter<bool>("hw_flow_control",  true);
        // 动作前必须配置的寄存器（控制器掉电后归零，不配置则动作指令有 ACK 但电机不转）
        speed_rpm_ = declare_parameter<int>("speed_rpm",  300);
        stroke_pulses_ = declare_parameter<int>("stroke_pulses",  30000);
        home_speed_rpm_ = declare_parameter<int>("home_speed_rpm",  200);
        // 无运动判定宽限: 现场实测控制器对动作命令有 ~3s "哑窗口"
        // (命令发出后 ~3s 才启动电机且期间不回 FC03), 1s 宽限会在电机
        // 启动前就误报 NO_MOTION, 故放宽到 4s。
        no_motion_grace_ms_ = declare_parameter<int>("no_motion_grace_ms", 4000);

        if (!rs485_.openPort(port_, baudrate_, hw_flow_control_))
        {
            RCLCPP_FATAL(get_logger(), "Failed to open RS485 port: %s", port_.c_str());
            throw std::runtime_error("Failed to open RS485 port.");
        }

        if (!configureDevice())
        {
            RCLCPP_ERROR(get_logger(),
                         "Scissor register init incomplete; motion commands may not move the motor.");
        }

        service_ =  create_service<SetScissorState>("/scissor/set_state",
                                                    std::bind(&ScissorControlNode::serviceCallback, this, std::placeholders::_1, std::placeholders::_2));
    }


private:

    /*
     * 启动时写一次速度(0x04)、行程(0x05)、回原点速度(0x1A)并各等 ACK。
     * 寄存器在控制器供电期间保持，节点每次启动配置一次即可。
     * 0x1A 掉电归零会导致回原点(开剪)不动 —— 必须配置。
     */
    bool configureDevice()
    {
        drainStale();

        const bool speed_ok = writeRegister(
            ScissorProtocol::REG_SPEED, static_cast<uint16_t>(speed_rpm_), 500);
        if (!speed_ok)
        {
            RCLCPP_ERROR(get_logger(), "Failed to write speed register 0x04 (no ACK)");
        }

        const bool stroke_ok = writeRegister(
            ScissorProtocol::REG_STROKE, static_cast<uint16_t>(stroke_pulses_), 500);
        if (!stroke_ok)
        {
            RCLCPP_ERROR(get_logger(), "Failed to write stroke register 0x05 (no ACK)");
        }

        const bool home_speed_ok = writeRegister(
            ScissorProtocol::REG_HOME_SPEED, static_cast<uint16_t>(home_speed_rpm_), 500);
        if (!home_speed_ok)
        {
            RCLCPP_ERROR(get_logger(), "Failed to write home speed register 0x1A (no ACK)");
        }

        return speed_ok && stroke_ok && home_speed_ok;
    }


    // 清空协议缓冲并排干串口残留数据（如上次动作迟到的停止帧）
    void drainStale()
    {
        protocol_.clearBuffer();
        std::vector<uint8_t> old_data;
        while (rs485_.receiveData(old_data, 256, 0) > 0)
        {
        }
    }


    // 读串口直到取出一帧或超时；接收出错置 rx_error
    bool waitForFrame(std::vector<uint8_t>& frame, int timeout_ms, bool& rx_error)
    {
        rx_error = false;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
        {
            std::vector<uint8_t> rx_data;
            const int count = rs485_.receiveData(rx_data, 256, 100);
            if (count < 0)
            {
                rx_error = true;
                return false;
            }
            if (count > 0)
            {
                protocol_.appendData(rx_data);
            }
            if (protocol_.tryPopFrame(frame))
            {
                return true;
            }
        }
        return false;
    }


    // 写单寄存器并等 ACK（原样回显），期间杂帧直接丢弃
    bool writeRegister(uint8_t reg, uint16_t value, int ack_timeout_ms)
    {
        const auto tx_frame = protocol_.buildWriteRegister(reg, value);
        RCLCPP_INFO(get_logger(), "TX: %s", bytesToHex(tx_frame).c_str());
        if (!rs485_.sendData(tx_frame))
        {
            return false;
        }

        bool rx_error = false;
        std::vector<uint8_t> frame;
        while (waitForFrame(frame, ack_timeout_ms, rx_error))
        {
            RCLCPP_INFO(get_logger(), "RX: %s", bytesToHex(frame).c_str());
            if (frame == tx_frame)
            {
                return true;
            }
        }
        return false;
    }


    // 读保持寄存器（FC03），期间杂帧（如停止帧）直接丢弃
    bool readRegister(uint8_t reg, uint16_t& value, int read_timeout_ms)
    {
        const auto tx_frame = protocol_.buildReadRegister(reg);
        if (!rs485_.sendData(tx_frame))
        {
            return false;
        }

        bool rx_error = false;
        std::vector<uint8_t> frame;
        while (waitForFrame(frame, read_timeout_ms, rx_error))
        {
            if (protocol_.parseReadResponse(frame, value))
            {
                return true;
            }
            RCLCPP_DEBUG(get_logger(), "discard frame: %s", bytesToHex(frame).c_str());
        }
        return false;
    }


    // 分片睡眠，期间响应 ROS 关停
    void sleepUntil(const std::chrono::steady_clock::time_point& wake_time)
    {
        while (rclcpp::ok() && std::chrono::steady_clock::now() < wake_time)
        {
            rclcpp::sleep_for(std::chrono::milliseconds(50));
        }
    }


    void serviceCallback(const std::shared_ptr<SetScissorState::Request> request, std::shared_ptr<SetScissorState::Response> response)
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        ScissorProtocol::Command command;
        std::string command_name;

        if (request->command == SetScissorState::Request::OPEN)
        {
            command = ScissorProtocol::Command::OPEN;
            command_name = "OPEN";
        }
        else if (request->command == SetScissorState::Request::CLOSE)
        {
            command = ScissorProtocol::Command::CLOSE;
            command_name = "CLOSE";
        }
        else
        {
            response->success = false;
            response->error_code = SetScissorState::Response::INVALID_COMMAND;
            response->message = "Invalid command.";
            return;
        }

        drainStale();

        const auto tx_frame = protocol_.buildCommand(command);
        RCLCPP_INFO(get_logger(), "TX %s: %s",
                    command_name.c_str(), bytesToHex(tx_frame).c_str());

        if (!rs485_.sendData(tx_frame))
        {
            response->success = false;
            response->error_code = SetScissorState::Response::SEND_FAILED;
            response->message = "RS485 send failed.";
            return;
        }

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);

        // 阶段1：等 ACK（写成功原样回显）。
        // 注意: 0x0A 回原点(开剪)命令实测控制器不回 ACK —— 电机照常执行,
        // 完成后只上发停止帧 01 06 00 03 00 01 B8 0A。因此 ACK 只短等,
        // 等不到不判失败, 命令是否生效以阶段3 的 0x02 运动状态为准。
        bool ack_received = false;
        bool rx_error = false;
        const auto ack_deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (rclcpp::ok() && std::chrono::steady_clock::now() < ack_deadline)
        {
            std::vector<uint8_t> frame;
            if (waitForFrame(frame, 100, rx_error))
            {
                RCLCPP_INFO(get_logger(), "RX: %s", bytesToHex(frame).c_str());
                if (frame == tx_frame)
                {
                    ack_received = true;
                    break;
                }
            }
            else if (rx_error)
            {
                response->success = false;
                response->error_code = SetScissorState::Response::RECEIVE_ERROR;
                response->message = "RS485 receive error.";
                return;
            }
        }
        if (!ack_received)
        {
            RCLCPP_WARN(get_logger(),
                        "No ACK within 500ms (0x0A 回原点不回 ACK 属正常), "
                        "以 0x02 运动状态判定结果");
        }

        // 阶段2：官方防呆建议，动作指令后先延时再读运动状态
        sleepUntil(std::chrono::steady_clock::now() + std::chrono::milliseconds(200));

        /*
         * 阶段3：轮询 0x02 运动状态。
         * 必须观察到 1（运动中）再回到 0（已停止）才算动作完成；
         * 停止帧 01 06 00 03 00 01 B8 0A 不可靠（可能迟到/残留/未动时不发），
         * 不作为完成判据，读到即丢弃。
         */
        bool seen_active = false;
        const auto grace_deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(no_motion_grace_ms_);
        while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
        {
            uint16_t motion_status = 0;
            if (readRegister(ScissorProtocol::REG_MOTION, motion_status, 100))
            {
                RCLCPP_INFO(get_logger(), "motion status 0x02 = %u", motion_status);
                if (motion_status == 1)
                {
                    seen_active = true;
                }
                else if (seen_active)
                {
                    response->success = true;
                    response->error_code = SetScissorState::Response::OK;
                    if (command == ScissorProtocol::Command::OPEN)
                    {
                        response->message = "Scissor opened successfully.";
                    }
                    else
                    {
                        response->message = "Scissor closed successfully.";
                    }
                    return;
                }
                else if (std::chrono::steady_clock::now() >= grace_deadline)
                {
                    response->success = false;
                    response->error_code = SetScissorState::Response::NO_MOTION;
                    response->message = "No motion detected within grace period "
                                        "(check speed/stroke registers or mechanical limits).";
                    return;
                }
            }
            // 读失败重读，不单独计时（与现场 Python 实现一致）
        }

        response->success = false;
        if (seen_active)
        {
            response->error_code = SetScissorState::Response::TIMEOUT;
            response->message = "Motion started but did not finish within timeout.";
        }
        else
        {
            response->error_code = SetScissorState::Response::NO_MOTION;
            response->message = "No motion detected within timeout "
                                "(check speed/stroke registers or mechanical limits).";
        }
    }


    std::string bytesToHex(const std::vector<uint8_t>& data) const
    {
        std::ostringstream oss;

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            if (i != 0)
            {
                oss << " ";
            }
            oss
                << std::hex
                << std::uppercase
                << std::setw(2)
                << std::setfill('0')
                << static_cast<int>(
                    data[i]);
        }
        return oss.str();
    }


private:
    RS485Serial rs485_;
    ScissorProtocol protocol_;
    std::mutex command_mutex_;
    std::string port_;
    int baudrate_;
    int timeout_ms_;
    bool hw_flow_control_;
    int speed_rpm_;
    int stroke_pulses_;
    int home_speed_rpm_;
    int no_motion_grace_ms_;

    rclcpp::Service<SetScissorState>::SharedPtr service_;
};


int main(int argc,  char** argv)
{
    rclcpp::init(argc, argv);
    auto node =std::make_shared<ScissorControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
