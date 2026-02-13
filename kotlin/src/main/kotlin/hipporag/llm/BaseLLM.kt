package hipporag.llm

import hipporag.utils.LlmResult
import hipporag.utils.Message

interface BaseLLM {
    fun infer(messages: List<Message>): LlmResult
}
