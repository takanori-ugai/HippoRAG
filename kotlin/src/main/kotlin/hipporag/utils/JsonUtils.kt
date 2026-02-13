package hipporag.utils

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonBuilder
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.contextual

val hippoSerializersModule: SerializersModule =
    SerializersModule {
        contextual(Any::class, AnyValueSerializer)
    }

fun jsonWithDefaults(configure: JsonBuilder.() -> Unit = {}): Json =
    Json {
        serializersModule = hippoSerializersModule
        configure()
    }
