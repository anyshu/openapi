# Model API Testing Framework

这是一个用于快速对接和测试新发布大语言模型的标准 API 测试框架。本项目提供了一个统一的接口，可以方便地将各种大语言模型转换为标准的 OpenAI API 格式进行测试和评估。

## 支持的模型

目前支持以下模型：
- THUDM/GLM-4
- Kimi Vision Language Model
- DREAM

## 安装

1. 克隆项目：
```bash
git clone <repository-url>
cd openapi
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动服务器：
```bash
python api.py --model-path "<model-path>" --model-name "<model-name>"
```

例如，启动 GLM-4 模型：
```bash
python api.py --model-path "THUDM/GLM-4-Z1-32B-0414" --model-name "glm-4"
```

2. 发送请求：

服务器启动后，可以通过 HTTP POST 请求访问 API：

```bash
curl http://localhost:12200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
  }'
```

## API 格式

API 遵循 OpenAI API 格式，主要端点：

- POST `/v1/chat/completions`：聊天补全接口

请求参数：
- `model`: 模型名称
- `messages`: 消息列表
- `temperature`：（可选）采样温度，默认 0
- `max_tokens`：（可选）最大生成长度，默认 4096

## 添加新模型

要添加新的模型支持，需要：

1. 在 `model_config.py` 中添加模型配置
2. 创建对应的模型处理器文件（参考 `glm4_handler.py`）
3. 实现必要的模型接口函数

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。