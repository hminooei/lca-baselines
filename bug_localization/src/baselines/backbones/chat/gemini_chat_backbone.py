from typing import Dict, Any, Optional, List
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.backbones.chat.prompts.chat_base_prompt import ChatBasePrompt
from src.baselines.utils.prompt_utils import batch_project_context, parse_list_files_completion
from src.baselines.utils.type_utils import ChatMessage

class GeminiChatBackbone(BaseBackbone):
    def __init__(
            self,
            name: str,
            model_name: str,
            prompt: ChatBasePrompt,
            parameters: Dict[str, Any],
            api_key: Optional[str] = None,
    ):
        super().__init__(name)
        genai.configure(api_key=api_key)
        self._model_name = model_name
        self._prompt = prompt
        self._parameters = parameters
        self._clint = genai.GenerativeModel(self._model_name)

    def _get_chat_completion(self, messages: List[ChatMessage]) -> str:
        chat = self._clint.start_chat(history=[])
        for message in messages:
            if message['role'] == 'user':
                chat.send_message(message['content'])
            elif message['role'] == 'assistant':
                # Simulate assistant messages in the chat history
                chat.history.append(genai.types.ContentType(role='model', parts=[message['content']]))
        
        response = chat.send_message(messages[-1]['content'])
        return response.text

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        batched_project_contents = batch_project_context(
            self._model_name, self._prompt, issue_description, repo_content, True
        )

        expected_files = set()
        raw_completions = []
        for batched_project_content in batched_project_contents:
            messages = self._prompt.chat(issue_description, batched_project_content)

            completion = self._get_chat_completion(messages)
            raw_completions.append(completion)

            expected_files.update(parse_list_files_completion(completion))

        return {
            "expected_files": list(expected_files),
            "raw_completions": raw_completions
        }
