package com.llmrouter.service;

import com.llmrouter.dto.R1Observation;
import com.llmrouter.r1.VerifiedR1PolicyBinding;
import com.llmrouter.repository.R1ObservationRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import tools.jackson.databind.json.JsonMapper;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import static com.llmrouter.dto.R1Observation.identifier;
import static com.llmrouter.dto.R1Observation.require;

/** Invoked only by a trusted Java evaluator/offline import adapter, never by chat completion. */
@Service
@ConditionalOnProperty(name = "llm.retraining.observation.enabled", havingValue = "true")
public class R1ObservationService {
    private static final Map<String, String> TIER_ACTIONS = Map.of("easy", "low", "medium", "medium", "hard", "high");
    private final R1ObservationRepository repository;
    private final JsonMapper jsonMapper;
    private final String targetContractVersion;

    public R1ObservationService(R1ObservationRepository repository, JsonMapper jsonMapper,
            @Value("${llm.retraining.observation.target-contract-version:}") String targetContractVersion) {
        identifier(targetContractVersion, "configured target contract version");
        this.repository = repository;
        this.jsonMapper = jsonMapper;
        this.targetContractVersion = targetContractVersion;
    }

    public R1Observation record(Assessment assessment, VerifiedR1PolicyBinding policy) {
        require(assessment != null && policy != null, "assessment and verified policy required");
        require(targetContractVersion.equals(assessment.targetContractVersion()), "target contract pin");
        var call = repository.findInvocation(assessment.requestId()).orElseThrow(() ->
                new IllegalArgumentException("R1 requires a recorded invocation"));
        require(assessment.requestId().equals(call.requestId()), "recorded request ID");
        require(!call.cacheHit() && "success".equals(call.status()) && call.fallbackReason() == null,
                "completed non-cache non-fallback invocation");
        require(call.origin() != null && Set.of("chat", "escalate").contains(call.origin()), "call origin");
        require("chat".equals(call.origin()) == call.requestId().equals(call.rootRequestId()), "root request ID");
        require(!call.rootCacheHit() && "success".equals(call.rootStatus()) && call.rootFallbackReason() == null,
                "completed non-cache non-fallback root invocation");
        require(call.question().equals(call.rootQuestion()), "root question provenance");
        require(policy.modelVersion().equals(call.routerModelVersion())
                && policy.modelVersion().equals(call.rootRouterModelVersion()), "verified router model version");
        String rootAction = TIER_ACTIONS.get(call.rootTier());
        String action = TIER_ACTIONS.get(call.tier());
        require(rootAction != null && action != null, "recorded tier projection");
        require(policy.candidateModelIds().get(rootAction).equals(call.rootModelId()), "actual root model ID");
        require(policy.candidateModelIds().get(action).equals(call.modelId())
                && call.modelId().equals(assessment.candidateModelId()), "assessed actual model ID");
        require(assessment.assessedAt() != null && !assessment.assessedAt().isBefore(call.recordedAt()),
                "assessment precedes invocation");
        var observation = new R1Observation(R1Observation.SCHEMA, assessment.eventId(), call.requestId(),
                assessment.redactedQuestion(), policy.modelVersion(), policy.policySha256(),
                policy.candidatePoolVersion(), call.modelId(), action, assessment.success(),
                assessment.assessmentSource(), assessment.assessmentId(), assessment.assessedAt(), assessment.piiRedacted());
        return repository.save(observation, call, policy.manifestSha256(), targetContractVersion,
                VerifiedR1PolicyBinding.sha256(call.question().getBytes(StandardCharsets.UTF_8)));
    }

    /** Produces a local immutable artifact body; this method performs no upload or training. */
    public String exportJson(List<String> eventIds) {
        return jsonMapper.writeValueAsString(new R1Observation.Batch(repository.findForExport(eventIds)));
    }

    /** The adapter supplies actual redaction and explicit assessment; booleans are never inferred. */
    public record Assessment(String eventId, UUID requestId, String candidateModelId,
                             @tools.jackson.databind.annotation.JsonDeserialize(using = R1Observation.StrictBoolean.class) Boolean success,
                             String assessmentSource, String assessmentId, Instant assessedAt,
                             String targetContractVersion, String redactedQuestion,
                             @tools.jackson.databind.annotation.JsonDeserialize(using = R1Observation.StrictBoolean.class) Boolean piiRedacted) {
        public Assessment {
            require(requestId != null, "request ID");
            R1Observation.eventIdentifier(eventId);
            identifier(candidateModelId, "candidate model ID");
            identifier(targetContractVersion, "target contract version");
            identifier(assessmentId, "assessment ID");
            require(success != null, "explicit success");
            require(assessedAt != null, "assessment time");
            require(Boolean.TRUE.equals(piiRedacted), "explicit PII redaction declaration");
            require(!R1Observation.blank(redactedQuestion), "redacted question");
            require(assessmentSource != null && Set.of("explicit_user", "reference_evaluator", "human_review")
                    .contains(assessmentSource), "explicit assessment source");
        }
    }
}
