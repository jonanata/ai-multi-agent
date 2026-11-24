# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    # TODO: Create a BedrockModel using the WORKER model
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1
    )

    # TODO: System prompt for the Inventory Agent
    system_prompt = (
        "You are the Inventory and Customer Data Agent for NovaMart. "
        "Your role is strictly to gather factual data from DynamoDB tables (orders and customer profiles). "
        "You must report raw facts accurately (order status, product name, order date, price, customer tier). "
        "You are a data gatherer only and must NEVER make return eligibility decisions or approve refunds."
    )

    # TODO: Implement check_order_status
    # NOTE: the Orders table has a COMPOSITE key (customer_id = partition key,
    # order_id = sort key), so a get_item needs BOTH values. That is why this
    # tool takes customer_id as well as order_id.
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price, ...)
            or a not-found message
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        try:
            response = table.get_item(Key={'customer_id': customer_id, 'order_id': order_id})
            return response.get('Item', {'error': f"Order {order_id} not found for customer {customer_id}"})
        except ClientError as e:
            return {'error': str(e)}

    # TODO: Implement get_customer_tier
    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        try:
            response = table.get_item(Key={'customer_id': customer_id})
            return response.get('Item', {'error': f"Customer {customer_id} not found"})
        except ClientError as e:
            return {'error': str(e)}

    # TODO: Implement list_customer_orders
    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        try:
            response = table.query(KeyConditionExpression=Key('customer_id').eq(customer_id))
            return {'orders': response.get('Items', [])}
        except ClientError as e:
            return {'error': str(e)}

    # TODO: Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders]
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    # TODO: Create a BedrockModel
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1
    )

    # TODO: System prompt for the Refund Agent
    system_prompt = (
        "You are the Refund and Return Eligibility Agent for NovaMart. "
        "Your task is to evaluate return and refund requests based on order facts retrieved by the Inventory Agent. "
        "Apply the correct return policy window according to the customer tier: Standard tier = 30 days, Premium tier = 60 days from the order date. "
        "Always check inventory context first using get_inventory_context before determining eligibility and initiating a refund."
    )

    # TODO: Implement get_inventory_context
    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id)
        if not state:
            return {}
        return state.get('inventory_agent', {})

    # TODO: Implement initiate_refund
    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        return_ref = f"RET-{uuid.uuid4().hex[:8].upper()}"
        try:
            table.update_item(
                Key={'customer_id': customer_id, 'order_id': order_id},
                UpdateExpression="SET #st = :st, return_reference = :ref, refund_reason = :reason",
                ExpressionAttributeNames={'#st': 'status'},
                ExpressionAttributeValues={':st': 'RefundInitiated', ':ref': return_ref, ':reason': reason},
                ReturnValues="UPDATED_NEW"
            )
            return {
                'status': 'success',
                'return_reference': return_ref,
                'message': f"Refund successfully initiated. Return reference number: {return_ref}"
            }
        except ClientError as e:
            return {'status': 'error', 'message': str(e)}

    # TODO: Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund]
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    # TODO: Build ReturnsPolicyRetrieverAgent
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        return retrieve_from_knowledge_base(query, config.RETURNS_KB_ID)

    # Create the ReturnsPolicyRetrieverAgent with the tool above
    returns_retriever = Agent(
        model=BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.0),
        system_prompt="You are a specialized returns policy retriever. Retrieve precise passages.",
        tools=[retrieve_returns_policy]
    )

    # TODO: Build ShippingPolicyRetrieverAgent
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        return retrieve_from_knowledge_base(query, config.SHIPPING_KB_ID)

    # Create the ShippingPolicyRetrieverAgent with the tool above
    shipping_retriever = Agent(
        model=BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.0),
        system_prompt="You are a specialized shipping policy retriever. Retrieve precise passages.",
        tools=[retrieve_shipping_policy]
    )

    # TODO: Build WarrantyPolicyRetrieverAgent
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        return retrieve_from_knowledge_base(query, config.WARRANTY_KB_ID)

    # Create the WarrantyPolicyRetrieverAgent with the tool above
    warranty_retriever = Agent(
        model=BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.0),
        system_prompt="You are a specialized warranty policy retriever. Retrieve precise passages.",
        tools=[retrieve_warranty_policy]
    )

    # TODO: Implement search_all_policies - parallel RAG retrieval tool
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        # Build a dict mapping domain names to their retriever agents
        retrievers = {
            'Returns': returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        # Define a helper to run one retriever sub-agent
        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).
            """
            return domain, agent(query)

        # Use ThreadPoolExecutor to run all three retrievers in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_run_retriever, domain, ag, query): domain for domain, ag in retrievers.items()}
            for future in as_completed(futures):
                domain, res = future.result()
                results[domain] = res

        # ── Trace: all KBs responded - print each result sequentially ─────
        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        # Combine results from all three domains and return
        return (
            f"--- Returns Policy ---\n{results.get('Returns', '[No results]')}\n\n"
            f"--- Shipping Policy ---\n{results.get('Shipping', '[No results]')}\n\n"
            f"--- Warranty Policy ---\n{results.get('Warranty', '[No results]')}"
        )

    # TODO: Create a BedrockModel for the PolicyAgent coordinator
    coordinator_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2
    )

    # TODO: System prompt for PolicyAgent coordinator
    system_prompt = (
        "You are the Policy Coordinator Agent for NovaMart. "
        "You answer policy questions by calling search_all_policies first to gather passages from returns, shipping, and warranty knowledge bases. "
        "Synthesize the retrieved passages into a complete, accurate, and grounded response addressing the customer's question."
    )

    # TODO: Instantiate and return the PolicyAgent coordinator
    return Agent(
        model=coordinator_model,
        system_prompt=system_prompt,
        tools=[search_all_policies]
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    # TODO: Create a BedrockModel
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.3
    )

    # TODO: System prompt for the Communication Agent
    system_prompt = (
        "You are the Communication Agent for NovaMart customer support. "
        "Your task is to read the complete WorkflowState using get_full_workflow_context and compose a warm, professional, empathetic, and clear final customer-facing response incorporating all gathered facts, policy details, or decisions."
    )

    # TODO: Implement get_full_workflow_context
    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        state = _read_workflow_state(session_id)
        return state or {}

    # TODO: Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context]
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    # TODO: Create a BedrockModel using the ORCHESTRATOR model
    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0
    )

    # TODO: System prompt for the Orchestrator
    # For arithmetic, skip Inventory, Policy and Refund, but still call
    # CommunicationAgent last. Round currency only after the full calculation.
    system_prompt = (
        "You are the Master Orchestrator Agent for NovaMart customer support. "
        "You manage the session lifecycle and route requests to appropriate specialist agents. "
        "Strictly adhere to these routing rules:\n"
        "1. Every request, always call initialize_session first.\n"
        "2. For order status, return, or refund requests, call route_to_inventory_agent first, then route_to_refund_agent.\n"
        "3. For policy meaning questions, call route_to_policy_agent.\n"
        "4. For account questions (e.g. customer tier), call route_to_inventory_agent (never route to policy agent).\n"
        "5. For math or calculation questions, skip Inventory, Policy, and Refund agents, but proceed directly to CommunicationAgent.\n"
        "6. Every request must always call route_to_communication_agent last to compose the final response. "
        "You must never write the final customer response yourself; always delegate to the communication agent as your last step."
    )

    # Each routing tool follows the same pattern:
    #   1. read the current WorkflowState  (_read_workflow_state)
    #   2. invoke the worker agent
    #   3. write its result back with optimistic locking
    #      (_update_workflow_state(session_id, {'<column>': text}, expected_version))
    # The terminal trace UI can show each step: call trace.step_start('inventory_agent')
    # before the worker runs and trace.step_done('inventory_agent', old_version) after.

    # TODO: Implement route_to_inventory_agent
    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        state = _read_workflow_state(session_id)
        version = int(state['version']) if state else 0
        trace.step_start('inventory_agent')
        result = inventory_agent(f"Customer ID: {customer_id}, Request: {request}")
        _update_workflow_state(session_id, {'inventory_agent': str(result)}, version)
        trace.step_done('inventory_agent', version)
        return str(result)

    # TODO: Implement route_to_policy_agent
    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _read_workflow_state(session_id)
        version = int(state['version']) if state else 0
        trace.step_start('policy_agent')
        result = policy_agent(request)
        _update_workflow_state(session_id, {'policy_agent': str(result)}, version)
        trace.step_done('policy_agent', version)
        return str(result)

    # TODO: Implement route_to_refund_agent
    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        state = _read_workflow_state(session_id)
        version = int(state['version']) if state else 0
        trace.step_start('refund_agent')
        result = refund_agent(f"Customer ID: {customer_id}, Request: {request}, Session ID: {session_id}")
        _update_workflow_state(session_id, {'refund_agent': str(result)}, version)
        trace.step_done('refund_agent', version)
        return str(result)

    # TODO: Implement route_to_communication_agent
    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        state = _read_workflow_state(session_id)
        version = int(state['version']) if state else 0
        trace.step_start('communication_agent')
        result = communication_agent(f"Session ID: {session_id}, Customer ID: {customer_id}, Original Request: {original_request}")
        _update_workflow_state(session_id, {'communication_agent': str(result)}, version)
        trace.step_done('communication_agent', version)
        return str(result)

    # TODO: Implement initialize_session
    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
            return f"Session {session_id} initialized successfully for customer {customer_id}."
        except Exception:
            return f"Session {session_id} already exists."

    # TODO: Instantiate and return the OrchestratorAgent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent
        ]
    )