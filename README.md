# Enterprise Multi-Agent Architecture with Amazon Bedrock AgentCore

*Project 3, Multi-Agent Support System — Future AWS Agent Engineer Nanodegree*

---



## Overview  

A production-grade multi-agent customer support system built with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python) and **Amazon Bedrock AgentCore**. The system routes customer support requests through an **Orchestrator → Workers** hierarchy, executes parallel multi-agent Retrieval-Augmented Generation (RAG) across domain Knowledge Bases backed by **Amazon S3 Vectors**, enforces enterprise safety policies via **Bedrock Guardrails**, maintains shared session state in **DynamoDB** with optimistic locking, and streams telemetry to **CloudWatch Logs** and **AWS X-Ray**.

### 📚 Detailed Articles & References

For a deep dive into the engineering principles, architectural decisions, and step-by-step walkthrough, check out the following resources:

* 🌐 **Technical Blog Post:** [Read full article](https://blog.jonanata.com/building-a-production-multi-agent-customer-support-system-with-strands-agents-sdk-and-amazon-bedrock-agentcore/)

* ☁️ **AWS Community Builders Article:** [Read on AWS Builders Center](https://builder.aws.com/content/3LvNEFpOuBTNAYB0cyMRGhcP8y6/autonomous-tool-orchestration-and-multi-capability-agent-systems-with-amazon-bedrock-agentcore-and-strands-sdk)

* 🔗 **Official Starter Template:** [udacity/cd14764-aws-agentic-c3-classroom/tree/main/project/starter](https://github.com/udacity/cd14764-aws-agentic-c3-classroom/tree/main/project/starter)

---

## Architecture

```text
Customer Request
      │
OrchestratorAgent (Claude Haiku 4.5 - routes requests, manages WorkflowState)
      │
   ┌──┼──────────────────────────┬──────────────────┐
   │  │                          │                  │
InventoryAgent    PolicyAgent    RefundAgent    CommunicationAgent
(DynamoDB)     (Multi-Agent RAG)  (DynamoDB)     (composes response)
                     │
         ┌───────────┼───────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
    (Bedrock KB)          (Bedrock KB)              (Bedrock KB)
         └───────── run in PARALLEL ───────────┘

Shared State: DynamoDB WorkflowStateTable (with optimistic locking)
```

### Agent Roles

| Agent | Model | Responsibility | Tools |
|-------|-------|-----------------|-------|
| **OrchestratorAgent** | Claude Haiku 4.5 | Routes requests, creates/updates `WorkflowState` | `initialize_session`, `route_to_inventory_agent`, `route_to_policy_agent`, `route_to_refund_agent`, `route_to_communication_agent` |
| **InventoryAgent** | Claude Sonnet 4.5 | Gathers order and customer facts from DynamoDB (no eligibility decisions) | `check_order_status`, `get_customer_tier`, `list_customer_orders` |
| **PolicyAgent** | Claude Sonnet 4.5 | Coordinates parallel RAG retrieval from 3 Knowledge Bases, synthesizes results | `search_all_policies` *(internally fans out to 3 retriever sub-agents)* |
| **RefundAgent** | Claude Sonnet 4.5 | Makes return/refund eligibility decisions (30-day Standard, 60-day Premium) | `get_inventory_context`, `initiate_refund` |
| **CommunicationAgent** | Claude Sonnet 4.5 | Reads full `WorkflowState` and composes final empathetic customer-facing response | `get_full_workflow_context` |

### Key Features

- **Tiered Model Routing**: Uses **Claude Haiku 4.5** in `OrchestratorAgent` for fast, low-cost routing and state updates, paired with **Claude Sonnet 4.5** in domain workers for reasoning, policy synthesis, and response composition.
- **Parallel Multi-Agent RAG**: `PolicyAgent` runs three specialized retriever sub-agents (`ReturnsPolicyRetriever`, `ShippingPolicyRetriever`, `WarrantyPolicyRetriever`) concurrently via `ThreadPoolExecutor` and synthesizes retrieved passages.
- **Shared Workflow State with Optimistic Locking**: DynamoDB `WorkflowStateTable` (keyed by `session_id`) tracks accumulated findings (`version: 0` through `version: N`) with conditional writes (`ConditionExpression='version = :expected_version'`) and retries on version conflicts.
- **Enterprise Guardrails**: Amazon Bedrock Guardrails enforce content filters (`SEXUAL`, `VIOLENCE`, `HATE` at `HIGH`; `INSULTS`, `MISCONDUCT` at `MEDIUM`), PII handling (`BLOCK` credit cards/SSNs; `ANONYMIZE` email/phone), denied topics (`CompetitorProducts`, `PricingNegotiations`, `LegalThreats` on `STANDARD` tier with cross-region profile `us.guardrail.v1:0`), and managed profanity filtering under a published numbered version.
- **Session Memory**: AgentCore Memory (`configure_memory`) creates a `SESSION_SUMMARY` (`summaryMemoryStrategy`) resource with 7-day event retention (`eventExpiryDuration=7`).
- **Observability**: AgentCore Observability routes execution logs to CloudWatch Logs (`INFO` level) and distributed traces to AWS X-Ray (`100%` sampling rate / `1.0`) across the `NovaMart-Orchestrator → Worker → Knowledge Base` call chain.
- **Serverless Vector Storage**: Uses **Amazon S3 Vectors** (vector bucket + three indexes: `returns-policy-index`, `shipping-policy-index`, `warranty-policy-index`) with self-managed Bedrock Knowledge Bases and **Amazon Titan Embed Text v2** embeddings.

---

## Prerequisites

- **Python 3.12** (matches runtime environment)
- **AWS Account** with Amazon Bedrock access enabled (`us-east-1`)
- **IAM Permissions** for Bedrock, Bedrock AgentCore, DynamoDB, S3, S3 Vectors, CloudFormation, CloudWatch Logs, and AWS X-Ray
- **Node.js 20+** and [**uv**](https://docs.astral.sh/uv/getting-started/installation/) if using the AgentCore CLI (`@aws/agentcore`) for runtime packaging and deployment

---

## Setup

### 1. Deploy Foundation Infrastructure

Deploy `infrastructure/starter_stack.yaml` via AWS CloudFormation:

```bash
aws cloudformation deploy \
  --template-file starter_stack.yaml \
  --stack-name your-agentcore \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

Validate stack deployment status:

```bash
aws cloudformation describe-stacks \
  --stack-name your-agentcore \
  --query "Stacks[0].StackStatus" \
  --region us-east-1
```

Confirm the status returns `CREATE_COMPLETE`.

### 2. Seed Initial Data

Populate DynamoDB tables (`agentcore-orders`, `agentcore-customers`, `agentcore-workflow-state`) and upload policy documents under `policies/returns/`, `policies/shipping/`, and `policies/warranty/` in the policy S3 bucket:

```bash
python infrastructure/seed_data.py
aws s3 ls s3://$(python -c "import config; print(config.POLICY_BUCKET)")/policies/ --recursive
```

### 3. Configure Self-Managed Bedrock Knowledge Bases (S3 Vectors)

Inspect your stack's resource names using `python config.py`, then create three **Self-managed** Knowledge Bases in **Amazon Bedrock → Knowledge Bases** (do not select *Managed KB*):

| Setting | Returns KB | Shipping KB | Warranty KB |
|---------|------------|-------------|-------------|
| **Name** | `novamart-returns-policy-kb` | `novamart-shipping-policy-kb` | `novamart-warranty-policy-kb` |
| **S3 Data Source Prefix** | `policies/returns/` | `policies/shipping/` | `policies/warranty/` |
| **Embedding Model** | Amazon Titan Embed Text v2 | Amazon Titan Embed Text v2 | Amazon Titan Embed Text v2 |
| **Vector Store** | S3 Vectors → Use an existing vector bucket | S3 Vectors → Use an existing vector bucket | S3 Vectors → Use an existing vector bucket |
| **S3 Vectors Bucket** | Vector Bucket from `python config.py` (`udacity-agentcore-vectors-…`) | Vector Bucket from `python config.py` (`udacity-agentcore-vectors-…`) | Vector Bucket from `python config.py` (`udacity-agentcore-vectors-…`) |
| **Vector Index** | `returns-policy-index` | `shipping-policy-index` | `warranty-policy-index` |

Click **Sync** on each Knowledge Base after creation, then copy the resulting Knowledge Base IDs into `.env`:

```bash
RETURNS_KB_ID=<your-returns-kb-id>
SHIPPING_KB_ID=<your-shipping-kb-id>
WARRANTY_KB_ID=<your-warranty-kb-id>
```

---

## Project Structure

```text
starter/
├── config.py                     # Central configuration (CloudFormation exports + env vars)
├── infrastructure/
│   ├── starter_stack.yaml        # CloudFormation template (DynamoDB, S3, S3 Vectors, IAM, CloudWatch)
│   └── seed_data.py              # Seeds mock customers, orders, and S3 policy documents
├── src/
│   ├── agent_orchestrator.py     # Multi-agent graph, guardrails, AgentCore deploy, memory, observability
│   ├── agent_utils.py            # Terminal trace UI and agent formatting helpers
│   ├── agent_observability.py    # @tool X-Ray instrumentation, logging, and observability config helpers
│   ├── agentcore_cli.py          # AgentCore CLI staging, configuration, and deployment wrapper
│   └── bedrock_kb_retrieval.py   # Bedrock Knowledge Base retrieval helper
└── tests/
    └── test_agent.py             # Task-by-task validation suite (task2 through task6, all)
```

---

## Orchestration & Routing Rules

`OrchestratorAgent` delegates all work to specialist agents and enforces six deterministic routing rules:

| Rule | Trigger | Action |
|------|---------|--------|
| **1** | Every request, always | Call `initialize_session` first |
| **2** | Order status / return / refund requests | Call `route_to_inventory_agent` first, then `route_to_refund_agent` |
| **3** | Policy meaning questions (return windows, shipping rates, warranty terms) | Call `route_to_policy_agent` |
| **4** | Account questions (*"what is my tier?"*, *"am I premium?"*) | Call `route_to_inventory_agent` — never `route_to_policy_agent` |
| **5** | Math / calculation questions | Skip Inventory, Policy, and Refund; route to `CommunicationAgent` |
| **6** | Every request, always (last step) | Call `route_to_communication_agent` to compose the final response |

---

## Deployment & Operations

Run the full deployment pipeline from `src/agent_orchestrator.py`:

```bash
python src/agent_orchestrator.py deploy
```

This pipeline executes:
1. Builds all five Strands agents (`InventoryAgent`, `RefundAgent`, `PolicyAgent`, `CommunicationAgent`, `OrchestratorAgent`).
2. Creates and versions the Bedrock Guardrail (`create_guardrail()` → `create_guardrail_version()`).
3. Stages runtime code and deploys to Amazon Bedrock AgentCore Runtime (`deploy_to_agentcore_runtime()`).
4. Provisions session memory with a 7-day `SESSION_SUMMARY` strategy (`configure_memory()`).
5. Enables CloudWatch Logs (`INFO`) and AWS X-Ray tracing at `100%` sampling (`configure_observability()`).
6. Optionally registers Lambda targets on an AgentCore Gateway (`deploy_agentcore_gateway()`) if `ORDERS_FUNCTION`, `POLICY_FUNCTION`, or `CUSTOMERS_FUNCTION` are set.

### Local Testing, Chat & Runtime Invocation

```bash
# Run local test scenarios with X-Ray tracing
python src/agent_orchestrator.py test

# Launch interactive terminal chat
python src/agent_orchestrator.py chat

# Invoke deployed AgentCore Runtime
python src/agent_orchestrator.py invoke "What is the return policy for premium customers?" CUST-002

# Run HTTP server (used inside AgentCore Runtime)
python src/agent_orchestrator.py serve
```

---

## Validation (`tests/test_agent.py`)

Validate your implementation by task or across the entire suite using `tests/test_agent.py`:

```bash
python tests/test_agent.py task2    # Test multi-agent orchestration
python tests/test_agent.py task3    # Test AgentCore deployment + guardrails
python tests/test_agent.py task4    # Test memory
python tests/test_agent.py task5    # Test Bedrock Knowledge Base configuration + retrieval
python tests/test_agent.py task6    # Test observability
python tests/test_agent.py all      # Run all tests
```

| Test Suite | What It Validates |
|------------|-------------------|
| **`task2`** | Instantiates `InventoryAgent` (3 tools), `PolicyAgent` (1 tool), and `OrchestratorAgent` (5 routing tools); verifies Haiku vs. Sonnet model tiering; statically verifies `ThreadPoolExecutor` usage in `build_policy_agent()`. |
| **`task3`** | Confirms Bedrock Guardrail exists with expected content filter strengths, PII rules, three `DENY` topics, and profanity policy; verifies a published numbered `GUARDRAIL_VERSION`; checks AgentCore Runtime is `READY` (`PUBLIC` / `HTTP`) with required environment variables. |
| **`task4`** | Verifies AgentCore Memory resource matching `config.MEMORY_NAME` is `ACTIVE`, uses `SUMMARIZATION` (`SESSION_SUMMARY`), and has a 7-day `eventExpiryDuration`. |
| **`task5`** | Verifies `RETURNS_KB_ID`, `SHIPPING_KB_ID`, and `WARRANTY_KB_ID` are set, each KB is `ACTIVE` with `amazon.titan-embed-text-v2:0` and a `COMPLETE` ingestion job, and `search_all_policies()` returns combined passages. |
| **`task6`** | Verifies CloudWatch logging (`AGENT_LOG_GROUP`, `INFO` level, enabled) and X-Ray tracing (`1.0` sampling rate, CloudWatch Logs destination, 100% default indexing rule), and checks recent `NovaMart-Orchestrator` traces. |

To inspect distributed traces visually after running `python src/agent_orchestrator.py test` or `invoke`, open **AWS Console → CloudWatch → X-Ray traces → Service map** (or **X-Ray → Service map**) and select **Last 5 minutes** to view the call graph from `NovaMart-Orchestrator` through worker agents and Knowledge Bases.

---

## Key Concepts

### Optimistic Locking on `WorkflowState`

```python
table.update_item(
    Key={'session_id': session_id},
    UpdateExpression=update_expr,
    ConditionExpression='version = :expected_version',
    ExpressionAttributeValues=expr_values
)
```

### Parallel RAG Retrieval

```python
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(_run_retriever, domain, ag, query): domain
        for domain, ag in retrievers.items()
    }
    for future in as_completed(futures):
        domain, res = future.result()
        results[domain] = res
```

### Versioned Bedrock Guardrail

```python
version_response = bedrock_client.create_guardrail_version(
    guardrailIdentifier=guardrail_id,
    description="Production version"
)
guardrail_version = version_response['version']
```

---

## Environment Variables

```bash
AWS_REGION=us-east-1
PROJECT_NAME=udacity-agentcore
RETURNS_KB_ID=
SHIPPING_KB_ID=
WARRANTY_KB_ID=
AGENTCORE_RUNTIME_ARN=
GUARDRAIL_ID=
GUARDRAIL_VERSION=
ORDERS_FUNCTION=
POLICY_FUNCTION=
CUSTOMERS_FUNCTION=
```

---

## Troubleshooting

- **Missing CloudFormation resources**: Confirm stack `your-agentcore` / `udacity-agentcore` is in `CREATE_COMPLETE` in `us-east-1` and run `python infrastructure/seed_data.py`.
- **Incorrect DynamoDB order lookups**: `check_order_status` requires both `customer_id` (partition key) and `order_id` (sort key).
- **Guardrail false positives on math queries**: Ensure `topicPolicyConfig` uses `tierConfig={"tierName": "STANDARD"}`, `crossRegionConfig={"guardrailProfileIdentifier": "us.guardrail.v1:0"}`, and defines `PricingNegotiations` to exclude arithmetic calculations on specified prices and discounts.
- **No X-Ray traces visible**: Confirm `AGENT_TRACING_ENABLED=true`, `AGENT_TRACE_SAMPLING_RATE=1.0`, and IAM permissions for `xray:PutTraceSegments`, then check **Last 5 minutes** in the X-Ray Service Map.  

