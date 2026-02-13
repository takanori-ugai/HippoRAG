package hipporag.llm

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import hipporag.utils.LlmResult
import hipporag.utils.Message

class LangChainChatLLM(
    private val model: ChatModel,
) : BaseLLM {
    override fun infer(messages: List<Message>): LlmResult {
        val chatMessages = messages.map { toChatMessage(it) }
        val response = model.chat(chatMessages)
        val aiMessage = response.aiMessage()
        val responseText =
            when (aiMessage) {
                is AiMessage -> aiMessage.text()
                else -> aiMessage.text()
            }

        val metadata = mutableMapOf<String, Any>()
        val tokenUsage = response.tokenUsage()
        if (tokenUsage != null) {
            metadata["prompt_tokens"] = tokenUsage.inputTokenCount()
            metadata["completion_tokens"] = tokenUsage.outputTokenCount()
            metadata["total_tokens"] = tokenUsage.totalTokenCount()
        }

        return LlmResult(response = responseText ?: "", metadata = metadata)
    }

    private fun toChatMessage(message: Message): ChatMessage =
        when (message.role.lowercase()) {
            "system" -> SystemMessage.from(message.content)
            "assistant", "ai" -> AiMessage.from(message.content)
            "user" -> UserMessage.from(message.content)
            else -> throw IllegalArgumentException("Unsupported message role: '${message.role}'")
        }
}
