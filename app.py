from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
import asyncio

# 環境変数の読み込み
load_dotenv()

# OpenAI APIクライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="絵本生成API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ehon-generate-app-62032802168.asia-northeast1.run.app",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# レスポンスIDを保持する辞書（本番環境ではRedisなどを使用）
response_cache: Dict[str, str] = {}


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]


class Tool(BaseModel):
    type: str
    quality: Optional[str] = None
    size: Optional[str] = None
    moderation: Optional[str] = None


class ChatRequest(BaseModel):
    model: str = "gpt-4.1-nano"
    input: Union[str, List[ChatMessage]]
    previous_response_id: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    model: str = "gpt-4.1-nano"
    input: str
    tools: List[Tool]
    previous_response_id: Optional[str] = None


class ImageEditRequest(BaseModel):
    model: str = "gpt-4.1-mini"
    input: List[ChatMessage]
    tools: List[Tool]
    previous_response_id: Optional[str] = None


class ApiResponse(BaseModel):
    id: str
    output: List[Any]


def get_user_metadata(
    x_user_name: Optional[str] = Header(None, alias="X-User-Name"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
) -> Dict[str, str]:
    """ヘッダーからユーザー情報を取得し、メタデータとして返す"""
    metadata = {}
    print(f"受信したヘッダー: X-User-Name={x_user_name}, X-User-Email={x_user_email}, X-User-Id={x_user_id}")

    if x_user_name:
        metadata["user_name"] = x_user_name
    if x_user_email:
        metadata["user_email"] = x_user_email
    if x_user_id:
        metadata["user_id"] = x_user_id

    print(f"生成されたメタデータ: {metadata}")
    return metadata


@app.get("/")
async def root():
    return {"message": "絵本生成API サーバーが稼働中です"}


@app.post("/api/v1/chat")
async def chat(
    request: ChatRequest,
    x_user_name: Optional[str] = Header(None, alias="X-User-Name"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """テキストチャット専用エンドポイント"""
    try:
        if not client.api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI APIキーが設定されていません"
            )

        print(f"チャットリクエスト: {request}")
        metadata = get_user_metadata(x_user_name, x_user_email, x_user_id)
        print(f"ユーザー情報: {metadata}")
        return await send_chat_message(request, metadata)

    except Exception as e:
        print(f"チャットAPIエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"チャットリクエストに失敗しました: {str(e)}"
        )


@app.post("/api/v1/generate-image")
async def generate_image_endpoint(
    request: ImageGenerationRequest,
    x_user_name: Optional[str] = Header(None, alias="X-User-Name"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """画像生成専用エンドポイント"""
    try:
        if not client.api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI APIキーが設定されていません"
            )

        print(f"画像生成リクエスト: {request}")
        metadata = get_user_metadata(x_user_name, x_user_email, x_user_id)
        print(f"ユーザー情報: {metadata}")
        return await generate_image(request, metadata)

    except Exception as e:
        print(f"画像生成APIエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"画像生成リクエストに失敗しました: {str(e)}"
        )


@app.post("/api/v1/edit-image")
async def edit_image_endpoint(
    request: ImageEditRequest,
    x_user_name: Optional[str] = Header(None, alias="X-User-Name"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """画像編集専用エンドポイント"""
    try:
        if not client.api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI APIキーが設定されていません"
            )

        print(f"画像編集リクエスト: {request}")
        metadata = get_user_metadata(x_user_name, x_user_email, x_user_id)
        print(f"ユーザー情報: {metadata}")
        return await edit_image(request, metadata)

    except Exception as e:
        print(f"画像編集APIエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"画像編集リクエストに失敗しました: {str(e)}"
        )


async def send_chat_message(request: ChatRequest, metadata: Dict[str, str]):
    """テキストメッセージの処理"""
    try:
        # Responses APIを使用してテキストメッセージを処理
        response_params = {
            "model": request.model,
            "input": request.input,
            "truncation": "auto",
            "metadata": metadata
        }

        # previous_response_idがある場合は追加
        if request.previous_response_id:
            response_params["previous_response_id"] = request.previous_response_id

        response = await asyncio.to_thread(
            client.responses.create,
            **response_params
        )

        # レスポンスIDをキャッシュに保存
        response_cache[response.id] = response.id

        # レスポンスの内容を取得
        text_content = response.output[0].content[0].text

        return ApiResponse(
            id=response.id,
            output=[{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": text_content
                }]
            }]
        )

    except Exception as e:
        print(f"チャットメッセージエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"チャットメッセージの処理に失敗しました: {str(e)}"
        )


async def generate_image(request: ImageGenerationRequest, metadata: Dict[str, str]):
    """画像生成の処理"""
    try:
        print("generate_image関数開始")

        # Responses APIを使用して画像生成（公式ドキュメント準拠）
        response_params = {
            "model": request.model,
            "input": request.input,
            "tools": request.tools,
            "truncation": "auto",
            "instructions": "画像には絶対に文字をいれないでください。背景は透過しないでください。",
            "metadata": metadata
        }
        print(f"response_params: {response_params}")

        # previous_response_idがある場合は追加
        if request.previous_response_id:
            response_params["previous_response_id"] = request.previous_response_id

        print("OpenAI API呼び出し開始")
        response = await asyncio.to_thread(
            client.responses.create,
            **response_params
        )
        print(f"OpenAI APIレスポンス受信: {response}")

        # レスポンスIDをキャッシュに保存
        response_cache[response.id] = response.id

        print(f"response.output: {response.output}")

        # 画像生成呼び出しを取得
        image_data = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]

        # テキストメッセージを取得
        text_data = [
            {
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": content.text
                } for content in output.content
                if content.type == "output_text"]
            }
            for output in response.output
            if output.type == "message"
        ]

        # 画像データがある場合は画像を、ない場合はテキストを返す
        output_data = image_data if image_data else text_data

        print(f"出力データ: {output_data}")

        return ApiResponse(
            id=response.id,
            output=output_data
        )

    except Exception as e:
        print(f"generate_image関数内エラー: {str(e)}")
        print(f"エラータイプ: {type(e)}")
        import traceback
        print(f"トレースバック: {traceback.format_exc()}")

        # 安全ポリシーエラーをチェック
        error_str = str(e)
        if "moderation_blocked" in error_str or "safety system" in error_str:
            # 安全ポリシーエラーの場合、通常のテキストレスポンスとして返す
            # previous_response_idを使用（ない場合はデフォルトID生成）
            import uuid
            response_id = request.previous_response_id or f"resp_{str(uuid.uuid4()).replace('-', '')}"
            response_cache[response_id] = response_id

            return ApiResponse(
                id=response_id,
                output=[{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "安全ポリシーによりブロックされました。プロンプトを修正してください。"
                    }]
                }]
            )

        raise HTTPException(
            status_code=500,
            detail=f"画像生成に失敗しました: {str(e)}"
        )


async def edit_image(request: ImageEditRequest, metadata: Dict[str, str]):
    """画像編集の処理"""
    try:
        print("edit_image関数開始")
        print(f"リクエスト内容: {request}")

        # Responses APIを使用して画像編集（公式ドキュメント準拠）
        response_params = {
            "model": request.model,
            "input": request.input,
            "tools": request.tools,
            "truncation": "auto",
            "metadata": metadata
        }
        print(f"response_params: {response_params}")

        # previous_response_idがある場合は追加
        if request.previous_response_id:
            response_params["previous_response_id"] = request.previous_response_id

        print("OpenAI API呼び出し開始")
        response = await asyncio.to_thread(
            client.responses.create,
            **response_params
        )
        print(f"OpenAI APIレスポンス: {response}")
        print(f"レスポンス出力: {response.output}")

        # 各出力の詳細をログ出力
        for i, output in enumerate(response.output):
            print(f"出力 {i}: type={output.type}, content={getattr(output, 'content', 'なし')}")

        # レスポンスIDをキャッシュに保存
        response_cache[response.id] = response.id

        # 画像生成呼び出しを取得（公式ドキュメント準拠）
        image_generation_calls = [
            output
            for output in response.output
            if output.type == "image_generation_call"
        ]
        print(f"画像生成呼び出し数: {len(image_generation_calls)}")

        # 画像データを取得
        image_data = [output.result for output in image_generation_calls]

        # テキストメッセージを取得
        text_data = [
            {
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": content.text
                } for content in output.content
                if content.type == "output_text"]
            }
            for output in response.output
            if output.type == "message"
        ]

        # 画像データがある場合は画像を、ない場合はテキストを返す
        output_data = image_data if image_data else text_data

        print(f"画像編集データ: {output_data}")

        return ApiResponse(
            id=response.id,
            output=output_data
        )

    except Exception as e:
        print(f"画像編集エラー: {str(e)}")
        import traceback
        print(f"トレースバック: {traceback.format_exc()}")

        # 安全ポリシーエラーをチェック
        error_str = str(e)
        if "moderation_blocked" in error_str or "safety system" in error_str:
            # 安全ポリシーエラーの場合、通常のテキストレスポンスとして返す
            # previous_response_idを使用（ない場合はデフォルトID生成）
            import uuid
            response_id = request.previous_response_id or f"resp_{str(uuid.uuid4()).replace('-', '')}"
            response_cache[response_id] = response_id

            return ApiResponse(
                id=response_id,
                output=[{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "安全ポリシーによりブロックされました。プロンプトを修正してください。"
                    }]
                }]
            )

        raise HTTPException(
            status_code=500,
            detail=f"画像編集に失敗しました: {str(e)}"
        )


@app.get("/api/v1/responses/{response_id}")
async def get_response(
    response_id: str,
    x_user_name: Optional[str] = Header(None, alias="X-User-Name"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """指定されたレスポンスIDの詳細情報を取得"""
    try:
        if not client.api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI APIキーが設定されていません"
            )

        metadata = get_user_metadata(x_user_name, x_user_email, x_user_id)
        print(f"ユーザー情報: {metadata}")

        # Responses APIを使用してレスポンス詳細を取得
        response = await asyncio.to_thread(
            client.responses.retrieve,
            response_id
        )

        return response

    except Exception as e:
        print(f"レスポンス取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"レスポンスの取得に失敗しました: {str(e)}"
        )


@app.get("/api/v1/responses/{response_id}/input_items")
async def get_input_items(
    response_id: str,
    x_user_name: Optional[str] = Header(None, alias="X-User-Name"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """指定されたレスポンスIDの入力アイテムを取得"""
    try:
        if not client.api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI APIキーが設定されていません"
            )

        metadata = get_user_metadata(x_user_name, x_user_email, x_user_id)
        print(f"ユーザー情報: {metadata}")

        # Responses APIを使用して入力アイテムを取得
        response = await asyncio.to_thread(
            client.responses.input_items.list,
            response_id
        )

        # Pydanticオブジェクトを辞書に変換（model_dump使用）
        serialized_data = []
        for item in response.data:
            try:
                # Pydanticのmodel_dumpメソッドを使用
                item_dict = item.model_dump()
                serialized_data.append(item_dict)
            except Exception as e:
                print(f"アイテムシリアライゼーションエラー: {e}")
                # フォールバック：基本属性のみ
                serialized_data.append({
                    "id": item.id,
                    "type": item.type,
                    "status": getattr(item, 'status', 'unknown')
                })

        return {"data": serialized_data}

    except Exception as e:
        print(f"入力アイテム取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"入力アイテムの取得に失敗しました: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
