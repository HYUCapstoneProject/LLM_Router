package com.llmrouter.r1;

import com.llmrouter.dto.R1Observation;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;

import static com.llmrouter.dto.R1Observation.require;

/** Only a caller-pinned, approved immutable package can bind v1 modelVersion to a policy. */
public final class VerifiedR1PolicyBinding {
    private final String modelVersion;
    private final String manifestSha256;
    private final String policySha256;
    private final String candidatePoolVersion;
    private final Map<String, String> candidateModelIds;

    private VerifiedR1PolicyBinding(String modelVersion, String manifestSha256, String policySha256,
                                    String pool, Map<String, String> modelIds) {
        this.modelVersion = modelVersion;
        this.manifestSha256 = manifestSha256;
        this.policySha256 = policySha256;
        this.candidatePoolVersion = pool;
        this.candidateModelIds = Map.copyOf(modelIds);
    }

    public static VerifiedR1PolicyBinding fromPinnedPackage(Path manifestPath, Path policyPath,
                                                            String expectedManifestSha256) throws IOException {
        require(expectedManifestSha256 != null && expectedManifestSha256.matches("[0-9a-f]{64}"),
                "pinned package manifest SHA");
        byte[] manifestBytes = Files.readAllBytes(manifestPath);
        require(sha256(manifestBytes).equals(expectedManifestSha256), "package manifest SHA mismatch");
        JsonMapper mapper = JsonMapper.builder().build();
        JsonNode manifest = mapper.readTree(manifestBytes);
        require("model-package-v2".equals(text(manifest, "schema_version")), "v2 package");
        require("approved".equals(text(manifest, "gate_status")), "approved package");
        require("pointwise".equals(text(manifest, "objective")), "pointwise package");
        require("routing_policy.json".equals(text(manifest, "routing_policy_file")), "policy file");
        String policyFileSha = text(manifest, "routing_policy_sha256");
        require(policyFileSha.equals(text(manifest.get("files"), "routing_policy.json")), "policy manifest binding");
        byte[] policyBytes = Files.readAllBytes(policyPath);
        require(sha256(policyBytes).equals(policyFileSha), "policy file SHA mismatch");
        JsonNode policy = mapper.readTree(policyBytes);
        require("routing-policy-v2".equals(text(policy, "schema_version")), "policy schema");
        require("pointwise".equals(text(policy, "objective")), "pointwise policy");
        require("independent_success_probabilities".equals(text(policy, "score_semantics")), "R1 score semantics");
        String canonicalSha = text(policy, "policy_sha256");
        require(canonicalSha.matches("[0-9a-f]{64}"), "canonical policy SHA");
        var pool = policy.get("candidate_pool");
        Map<String, String> models = new LinkedHashMap<>();
        require(pool != null && pool.get("candidates") != null && pool.get("candidates").isArray(), "candidates");
        for (JsonNode candidate : pool.get("candidates")) {
            require(models.put(text(candidate, "action"), text(candidate, "model_id")) == null, "duplicate action");
        }
        require(models.keySet().equals(R1Observation.ACTIONS), "three candidate actions");
        require(models.values().stream().distinct().count() == 3, "distinct candidate models");
        return new VerifiedR1PolicyBinding(text(manifest, "model_version"), expectedManifestSha256,
                canonicalSha, text(pool, "version"), models);
    }

    private static String text(JsonNode node, String field) {
        String value = node == null || node.get(field) == null ? null : node.get(field).stringValue();
        R1Observation.identifier(value, field);
        return value;
    }
    public static String sha256(byte[] bytes) {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
        catch (NoSuchAlgorithmException error) { throw new IllegalStateException(error); }
    }
    public String modelVersion() { return modelVersion; }
    public String manifestSha256() { return manifestSha256; }
    public String policySha256() { return policySha256; }
    public String candidatePoolVersion() { return candidatePoolVersion; }
    public Map<String, String> candidateModelIds() { return candidateModelIds; }
}
