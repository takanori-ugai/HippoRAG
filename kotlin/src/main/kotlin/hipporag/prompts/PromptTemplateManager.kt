package hipporag.prompts

import hipporag.utils.Message
import io.github.oshai.kotlinlogging.KotlinLogging

class PromptTemplateManager(
    private val roleMapping: Map<String, String>,
) {
    private val logger = KotlinLogging.logger {}
    private val templates: Map<String, PromptTemplate> = PromptTemplates.templates

    fun render(
        name: String,
        promptUser: String,
    ): List<Message> = render(name, mapOf("prompt_user" to promptUser))

    fun render(
        name: String,
        variables: Map<String, String>,
    ): List<Message> {
        val template = templates[name] ?: throw IllegalArgumentException("Template '$name' not found.")
        return when (template) {
            is PromptTemplate.Text -> {
                listOf(
                    Message(role = roleMapping["user"] ?: "user", content = substitute(template.template, variables)),
                )
            }

            is PromptTemplate.Chat -> {
                template.messages.map { msg ->
                    val role = roleMapping[msg.role] ?: msg.role
                    Message(role = role, content = substitute(msg.content, variables))
                }
            }
        }
    }

    fun listTemplateNames(): List<String> = templates.keys.toList()

    fun isTemplateNameValid(name: String): Boolean = templates.containsKey(name)

    private fun substitute(
        template: String,
        variables: Map<String, String>,
    ): String {
        val pattern = Regex("""\$\{(\w+)\}""")
        return pattern.replace(template) { match ->
            val key = match.groupValues[1]
            variables[key] ?: run {
                logger.warn { "Missing prompt variable '$key'." }
                match.value
            }
        }
    }
}

sealed class PromptTemplate {
    data class Text(
        val template: String,
    ) : PromptTemplate()

    data class Chat(
        val messages: List<PromptMessageTemplate>,
    ) : PromptTemplate()
}

data class PromptMessageTemplate(
    val role: String,
    val content: String,
)

object PromptTemplates {
    val templates: Map<String, PromptTemplate> = mapOf()
}
