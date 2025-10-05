import uvicorn
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel



# 导入 SSE GTPlanner API
from agent.api.agent_api import SSEGTPlanner

# 导入索引管理器
from agent.utils.startup_init import initialize_application

# 导入导出器
from agent.utils.export_planner import PlannerExporter
from agent.persistence.sqlite_session_manager import SQLiteSessionManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GTPlanner API",
    description="智能规划助手 API，支持流式响应和实时工具调用",
    version="1.0.0"
)

# 应用启动事件 - 预加载工具索引
@app.on_event("startup")
async def startup_event():
    """应用启动时预加载工具索引"""
    logger.info("🚀 GTPlanner API 启动中...")

    try:
        # 初始化应用，包括预加载工具索引
        result = await initialize_application(
            tools_dir="tools",
            preload_index=True
        )

        if result["success"]:
            logger.info("✅ 应用初始化成功")
            if "tool_index" in result["components"]:
                index_info = result["components"]["tool_index"]
                logger.info(f"📋 工具索引已就绪: {index_info.get('index_name', 'N/A')}")
        else:
            logger.error("❌ 应用初始化失败")
            for error in result["errors"]:
                logger.error(f"  - {error}")

    except Exception as e:
        logger.error(f"❌ 启动时初始化失败: {str(e)}")
        # 不阻止应用启动，但记录错误

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 现有路由已移除，只保留 SSE Agent 路由

# 创建全局 SSE API 实例
sse_api = SSEGTPlanner(verbose=True)

# 创建全局会话管理器和导出器
session_manager = SQLiteSessionManager()
exporter = PlannerExporter(session_manager)

# 请求模型
class AgentContextRequest(BaseModel):
    """AgentContext 请求模型（直接对应后端 AgentContext）"""
    session_id: str
    dialogue_history: List[Dict[str, Any]]
    tool_execution_results: Dict[str, Any] = {}
    session_metadata: Dict[str, Any] = {}
    last_updated: Optional[str] = None
    is_compressed: bool = False
    language: Optional[str] = None  # 新增：语言选择字段，支持 'zh', 'en', 'ja', 'es', 'fr'

    # SSE 配置选项（不属于 AgentContext，但用于 API 配置）
    include_metadata: bool = False
    buffer_events: bool = False
    heartbeat_interval: float = 30.0

# 健康检查端点（增强版）
@app.get("/health")
async def health_check():
    """增强的健康检查端点，包含 API 状态信息"""
    api_status = sse_api.get_api_status()
    return {
        "status": "healthy",
        "service": "gtplanner",
        "timestamp": datetime.now().isoformat(),
        "api_status": api_status
    }

@app.get("/api/status")
async def api_status():
    """获取详细的 API 状态信息"""
    return sse_api.get_api_status()

# 测试页面端点已移除

# 普通聊天API已移除，只保留SSE Agent API

@app.post("/api/chat/agent")
async def chat_agent_stream(request: AgentContextRequest):
    """SSE 流式聊天端点 - GTPlanner Agent"""
    try:
        # 验证 AgentContext 数据
        if not request.session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")

        if not request.dialogue_history:
            raise HTTPException(status_code=400, detail="dialogue_history cannot be empty")

        logger.info(f"Starting SSE stream for session: {request.session_id}, messages: {len(request.dialogue_history)}")

        async def generate_sse_stream():
            """生成 SSE 数据流"""
            try:
                # 发送连接建立事件
                connection_event = {
                    "status": "connected",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": request.session_id,
                    "dialogue_history_length": len(request.dialogue_history),
                    "config": {
                        "include_metadata": request.include_metadata,
                        "buffer_events": request.buffer_events,
                        "heartbeat_interval": request.heartbeat_interval
                    }
                }
                yield f"event: connection\ndata: {json.dumps(connection_event, ensure_ascii=False)}\n\n"

                # 创建一个队列来收集 SSE 数据
                import asyncio
                sse_queue = asyncio.Queue()
                processing_complete = False

                async def queue_sse_data(data: str):
                    """将 SSE 数据放入队列"""
                    await sse_queue.put(data)

                # 启动处理任务
                async def process_request():
                    nonlocal processing_complete
                    try:
                        # 构建 AgentContext 数据（移除冗余的 user_input）
                        agent_context = {
                            "session_id": request.session_id,
                            "dialogue_history": request.dialogue_history,
                            "tool_execution_results": request.tool_execution_results,
                            "session_metadata": request.session_metadata,
                            "last_updated": request.last_updated,
                            "is_compressed": request.is_compressed
                        }

                        language = request.session_metadata.get('language', 'zh')

                        result = await sse_api.process_request_stream(
                            agent_context=agent_context,
                            language=language,  # 作为独立参数传递语言选择
                            response_writer=queue_sse_data,
                            include_metadata=request.include_metadata,
                            buffer_events=request.buffer_events,
                            heartbeat_interval=request.heartbeat_interval
                        )

                        # 发送完成事件
                        completion_event = {
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        }
                        await sse_queue.put(f"event: complete\ndata: {json.dumps(completion_event, ensure_ascii=False)}\n\n")

                        # 发送连接关闭事件
                        close_event = {
                            "status": "closing",
                            "message": "Stream completed successfully",
                            "timestamp": datetime.now().isoformat()
                        }
                        await sse_queue.put(f"event: close\ndata: {json.dumps(close_event, ensure_ascii=False)}\n\n")

                        logger.info(f"SSE stream completed successfully for session: {result.get('session_id', 'unknown')}")

                    except Exception as e:
                        logger.error(f"SSE processing error: {e}", exc_info=True)
                        # 发送错误事件
                        error_event = {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "timestamp": datetime.now().isoformat()
                        }
                        await sse_queue.put(f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n")
                    finally:
                        processing_complete = True
                        await sse_queue.put(None)  # 结束标记

                # 启动处理任务
                task = asyncio.create_task(process_request())

                # 从队列中读取并发送数据
                while True:
                    try:
                        # 等待数据，设置超时避免无限等待
                        data = await asyncio.wait_for(sse_queue.get(), timeout=1.0)
                        if data is None:  # 结束标记
                            break
                        yield data
                    except asyncio.TimeoutError:
                        # 发送心跳保持连接
                        if not processing_complete:
                            heartbeat = f"event: heartbeat\ndata: {{\"timestamp\": \"{datetime.now().isoformat()}\"}}\n\n"
                            yield heartbeat
                        else:
                            break

                # 确保任务完成
                if not task.done():
                    await task

            except Exception as e:
                logger.error(f"SSE stream error: {e}", exc_info=True)
                # 发送错误事件
                error_event = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            }
        )

    except Exception as e:
        logger.error(f"Chat agent stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 导出端点
class ExportRequest(BaseModel):
    """导出请求模型"""
    session_id: str
    include_conversation: bool = True
    language: str = "zh"

@app.post("/api/export/markdown")
async def export_to_markdown(request: ExportRequest):
    """
    导出会话规划到 Markdown 文件

    参数:
    - session_id: 会话ID
    - include_conversation: 是否包含完整对话历史（默认 True）
    - language: 输出语言 (zh/en)

    返回:
    - file_path: 保存的文件路径
    - content: Markdown 内容
    """
    try:
        # 验证会话是否存在
        from agent.persistence.database_dao import DatabaseDAO
        dao = DatabaseDAO()
        session_info = dao.get_session(request.session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id}")

        # 创建导出器（使用请求的语言）
        exporter_instance = PlannerExporter(session_manager, language=request.language)

        # 导出到文件（不指定output_path，让系统自动生成文件名）
        file_path = exporter_instance.export_session_to_markdown(
            session_id=request.session_id,
            include_conversation=request.include_conversation
        )

        # 读取生成的内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.info(f"Exported session {request.session_id} to {file_path}")

        return {
            "success": True,
            "file_path": file_path,
            "content": content,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/preview/{session_id}")
async def preview_markdown(
    session_id: str,
    language: str = Query("zh", description="输出语言 (zh/en)"),
    max_length: int = Query(1000, description="最大预览长度")
):
    """
    预览会话规划的 Markdown 内容（不保存文件）

    参数:
    - session_id: 会话ID
    - language: 输出语言
    - max_length: 最大预览长度

    返回:
    - preview: Markdown 预览内容
    """
    try:
        # 验证会话是否存在
        from agent.persistence.database_dao import DatabaseDAO
        dao = DatabaseDAO()
        session_info = dao.get_session(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        # 创建导出器（使用请求的语言）
        exporter_instance = PlannerExporter(session_manager, language=language)

        # 获取预览
        preview_content = exporter_instance.get_markdown_preview(
            session_id=session_id,
            max_length=max_length
        )

        return {
            "success": True,
            "preview": preview_content,
            "session_id": session_id,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("fastapi_main:app", host="0.0.0.0", port=11211, reload=True)
