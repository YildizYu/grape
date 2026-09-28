#include "grasp_harvest_control/scissor_protocol.hpp"


std::vector<uint8_t> ScissorProtocol::buildCommand(
    Command command) const
{
    if (command == Command::CLOSE)
    {
        // 合上剪刀: 0x01 写 4 = 电机轴伸出合剪
        // 01 06 00 01 00 04 D9 C9
        return buildWriteRegister(0x01, 4);
    }

    // 打开剪刀: 0x0A 写 1 = 回原点, 电机轴缩回开剪
    // 01 06 00 0A 00 01 68 08
    return buildWriteRegister(0x0A, 1);
}


std::vector<uint8_t> ScissorProtocol::buildWriteRegister(
    uint8_t reg,
    uint16_t value) const
{
    std::vector<uint8_t> frame = {
        0x01,
        0x06,
        static_cast<uint8_t>((reg >> 8) & 0xFF),
        static_cast<uint8_t>(reg & 0xFF),
        static_cast<uint8_t>((value >> 8) & 0xFF),
        static_cast<uint8_t>(value & 0xFF)
    };

    const uint16_t crc =
        calculateCRC(frame.data(), frame.size());

    // Modbus RTU CRC: 低字节在前
    frame.push_back(static_cast<uint8_t>(crc & 0xFF));
    frame.push_back(static_cast<uint8_t>((crc >> 8) & 0xFF));
    return frame;
}


std::vector<uint8_t> ScissorProtocol::buildReadRegister(
    uint8_t reg,
    uint8_t count) const
{
    std::vector<uint8_t> frame = {
        0x01,
        0x03,
        static_cast<uint8_t>((reg >> 8) & 0xFF),
        static_cast<uint8_t>(reg & 0xFF),
        0x00,
        count
    };

    const uint16_t crc =
        calculateCRC(frame.data(), frame.size());

    frame.push_back(static_cast<uint8_t>(crc & 0xFF));
    frame.push_back(static_cast<uint8_t>((crc >> 8) & 0xFF));
    return frame;
}


void ScissorProtocol::appendData(
    const std::vector<uint8_t>& data)
{
    if (data.empty()) {
        return;
    }

    rx_buffer_.insert(
        rx_buffer_.end(),
        data.begin(),
        data.end());
}


bool ScissorProtocol::tryPopFrame(
    std::vector<uint8_t>& frame)
{
    frame.clear();

    /*
     * 当前协议正常帧：
     *
     * FC06 写响应   01 06 XX XX XX XX CRC CRC   (8字节)
     * FC03 读响应   01 03 02 XX XX CRC CRC      (7字节, 1个寄存器)
     */

    while (rx_buffer_.size() >= 7)
    {
        // 从机地址必须是 0x01
        if (rx_buffer_[0] != 0x01)
        {
            rx_buffer_.erase(
                rx_buffer_.begin());

            continue;
        }

        std::size_t frame_len = 0;
        if (rx_buffer_[1] == 0x06)
        {
            frame_len = 8;
        }
        else if (rx_buffer_[1] == 0x03)
        {
            frame_len = 7;
        }
        else
        {
            // 未知功能码(如干扰帧), 丢弃一个字节继续寻找
            rx_buffer_.erase(
                rx_buffer_.begin());

            continue;
        }

        // 帧尚未收全, 等待更多字节
        if (rx_buffer_.size() < frame_len)
        {
            return false;
        }

        std::vector<uint8_t> candidate(
            rx_buffer_.begin(),
            rx_buffer_.begin() + frame_len);


        // 检查CRC
        if (!checkCRC(candidate))
        {
            /*
             * 当前字节不是一帧的真正开始位置，
             * 删除一个字节后继续寻找。
             */
            rx_buffer_.erase(
                rx_buffer_.begin());

            continue;
        }


        // 找到一帧完整有效的数据
        frame = candidate;


        // 删除已经解析的数据
        rx_buffer_.erase(
            rx_buffer_.begin(),
            rx_buffer_.begin() + frame_len);

        return true;
    }
    return false;
}


bool ScissorProtocol::isCommandAck(
    const std::vector<uint8_t>& frame,
    Command command) const
{
    /*
     * 0x06 写单个寄存器正常响应
     * 与请求报文完全相同。
     */
    const auto expected =
        buildCommand(command);

    return frame == expected;
}


bool ScissorProtocol::isMotionComplete(
    const std::vector<uint8_t>& frame) const
{
    /*
     * 剪刀动作完成后控制器可能主动上发的停止帧：
     *
     * 01 06 00 03 00 01 B8 0A
     *
     * 注意: 该帧不可靠(可能迟到/残留/电机未动时不发),
     * 动作完成判定应以轮询 0x02 运动状态为准。
     */
    static const std::vector<uint8_t> complete_frame = {
        0x01,
        0x06,
        0x00,
        0x03,
        0x00,
        0x01,
        0xB8,
        0x0A
    };

    return frame == complete_frame;
}


bool ScissorProtocol::parseReadResponse(
    const std::vector<uint8_t>& frame,
    uint16_t& value) const
{
    /*
     * FC03 读响应: 01 03 02 XX XX CRC CRC
     * 帧已由 tryPopFrame 通过 CRC 校验。
     */
    if (frame.size() != 7 ||
        frame[0] != 0x01 ||
        frame[1] != 0x03 ||
        frame[2] != 0x02)
    {
        return false;
    }

    value =
        static_cast<uint16_t>(frame[3] << 8)
        |
        static_cast<uint16_t>(frame[4]);

    return true;
}


void ScissorProtocol::clearBuffer()
{
    rx_buffer_.clear();
}


uint16_t ScissorProtocol::calculateCRC(
    const uint8_t* data,
    std::size_t length) const
{
    uint16_t crc = 0xFFFF;

    for (std::size_t i = 0;
         i < length;
         ++i)
    {
        crc ^= data[i];

        for (int j = 0;
             j < 8;
             ++j)
        {
            if (crc & 0x0001)
            {
                crc >>= 1;
                crc ^= 0xA001;
            }
            else
            {
                crc >>= 1;
            }
        }
    }

    return crc;
}


bool ScissorProtocol::checkCRC(
    const std::vector<uint8_t>& frame) const
{
    // FC06 帧 8 字节, FC03 单寄存器响应 7 字节
    if (frame.size() != 7 && frame.size() != 8) {
        return false;
    }

    /*
     * 除 CRC 外的所有字节参与计算
     */
    uint16_t calculated =
        calculateCRC(
            frame.data(),
            frame.size() - 2);


    /*
     * Modbus RTU CRC：
     *
     * 低字节在前
     * 高字节在后
     */
    uint16_t received =
        static_cast<uint16_t>(frame[frame.size() - 2])
        |
        (
            static_cast<uint16_t>(frame[frame.size() - 1])
            << 8
        );


    return calculated == received;
}