package com.llmrouter.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import tools.jackson.core.JsonParser;
import tools.jackson.core.JsonToken;
import tools.jackson.databind.DeserializationContext;
import tools.jackson.databind.ValueDeserializer;
import tools.jackson.databind.annotation.JsonDeserialize;

import java.time.Instant;
import java.util.List;
import java.util.Set;
import java.util.UUID;

/** Internal artifact contract; no HTTP controller accepts or emits this DTO. */
public record R1Observation(
        @JsonProperty("schema_version") String schemaVersion,
        @JsonProperty("event_id") String eventId,
        @JsonProperty("request_id") UUID requestId,
        String question,
        @JsonProperty("router_model_version") String routerModelVersion,
        @JsonProperty("policy_sha256") String policySha256,
        @JsonProperty("candidate_pool_version") String candidatePoolVersion,
        @JsonProperty("candidate_model_id") String candidateModelId,
        String action,
        @JsonDeserialize(using = StrictBoolean.class) Boolean success,
        @JsonProperty("assessment_source") String assessmentSource,
        @JsonProperty("assessment_id") String assessmentId,
        @JsonProperty("assessed_at") Instant assessedAt,
        @JsonProperty("pii_redacted") @JsonDeserialize(using = StrictBoolean.class) Boolean piiRedacted
) {
    public static final String SCHEMA = "r1-observation-v1";
    public static final Set<String> ACTIONS = Set.of("low", "medium", "high");
    public R1Observation {
        require(SCHEMA.equals(schemaVersion), "schema_version");
        eventIdentifier(eventId);
        require(requestId != null, "request_id");
        require(!blank(question), "question");
        identifier(routerModelVersion, "router_model_version");
        require(policySha256 != null && policySha256.matches("[0-9a-f]{64}"), "policy_sha256");
        identifier(candidatePoolVersion, "candidate_pool_version");
        identifier(candidateModelId, "candidate_model_id");
        require(action != null && ACTIONS.contains(action), "action");
        require(success != null, "explicit success");
        require(assessmentSource != null && Set.of("explicit_user", "reference_evaluator", "human_review").contains(assessmentSource),
                "assessment_source");
        identifier(assessmentId, "assessment_id");
        require(assessedAt != null && assessedAt.toString().matches("[0-9]{4}-.*Z")
                && !assessedAt.toString().startsWith("0000-"), "assessed_at must be a supported UTC year");
        require(Boolean.TRUE.equals(piiRedacted), "explicit pii_redacted declaration");
    }

    public record Batch(@JsonProperty("schema_version") String schemaVersion,
                        List<R1Observation> observations) {
        public Batch(List<R1Observation> observations) { this("r1-observation-batch-v1", observations); }
        public Batch {
            require("r1-observation-batch-v1".equals(schemaVersion), "batch schema_version");
            observations = List.copyOf(observations);
            require(!observations.isEmpty(), "observations");
        }
    }

    public static class StrictBoolean extends ValueDeserializer<Boolean> {
        @Override
        public Boolean deserialize(JsonParser parser, DeserializationContext context) {
            if (parser.currentToken() == JsonToken.VALUE_TRUE) return true;
            if (parser.currentToken() == JsonToken.VALUE_FALSE) return false;
            return context.reportInputMismatch(Boolean.class, "explicit JSON boolean required");
        }
    }

    public static void eventIdentifier(String value) {
        require(value != null && value.matches("[A-Za-z0-9._-]{1,200}"), "event_id");
    }
    // Match Python str.isspace/strip, including Unicode NEL and nonbreaking spaces.
    private static boolean whitespace(int value) {
        return Character.isWhitespace(value) || Character.isSpaceChar(value) || value == 0x85;
    }
    public static boolean blank(String value) {
        return value == null || value.codePoints().allMatch(R1Observation::whitespace);
    }

    public static void identifier(String value, String name) {
        require(value != null && !value.isBlank() && value.length() <= 200
                && value.codePoints().noneMatch(R1Observation::whitespace), name);
    }
    public static void require(boolean value, String message) {
        if (!value) throw new IllegalArgumentException("R1 observation contract: " + message);
    }
}
