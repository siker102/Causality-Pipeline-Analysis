import streamlit as st
import pandas as pd
from causallearn.graph.Edge import Edge
from causallearn.search.ConstraintBased.FCI import fci
from typing import List,  Dict, Tuple
from collections import OrderedDict
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

# Reset function for when changing from generated data to uploaded data
def reset_background_knowledge_state():
    """Reset session state for background knowledge."""
    # Clear forbidden and required pairs
    for key in list(st.session_state.keys()):
        if key.endswith("_pairs"):
            del st.session_state[key]

    # Reset idp results state
    for key in list(st.session_state.keys()):
        if key.endswith("results"):
            del st.session_state[key]
    
    # Reset tier assignments 
    for key in list(st.session_state.keys()):
        if key.startswith("tier_"):
            st.session_state[key] = 5  # Set the base value for tiers

def _add_relationship_controls(df: pd.DataFrame, relationship_type: str) -> List[Tuple[str, str]]:
    """Create UI for adding edge relationships with proper state management."""
    cols = df.columns.tolist()
    
    # Initialize session state for tracking pairs
    if f"{relationship_type}_pairs" not in st.session_state:
        st.session_state[f"{relationship_type}_pairs"] = []
    
    with st.container():
        st.markdown(f"### {relationship_type.capitalize()} Relationships")
        
        # Display existing pairs with removal option
        for idx, pair in enumerate(st.session_state[f"{relationship_type}_pairs"]):
            col1, col2, col3 = st.columns([3, 3, 1])
            with col1:
                st.selectbox(
                    "From variable", 
                    [pair[0]], 
                    key=f"{relationship_type}_from_existing_{idx}",
                    disabled=True
                )
            with col2:
                st.selectbox(
                    "To variable", 
                    [pair[1]], 
                    key=f"{relationship_type}_to_existing_{idx}",
                    disabled=True
                )
            with col3:
                if st.button("❌", key=f"remove_{relationship_type}_{idx}"):
                    del st.session_state[f"{relationship_type}_pairs"][idx]
                    st.rerun()

        # Add new pair controls
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            new_from = st.selectbox(
                "From variable", 
                cols, 
                key=f"new_{relationship_type}_from"
            )
        with col2:
            new_to = st.selectbox(
                "To variable", 
                cols, 
                key=f"new_{relationship_type}_to"
            )
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add", key=f"add_{relationship_type}"):
                if{new_from != new_to}:
                    if (new_from, new_to) not in st.session_state[f"{relationship_type}_pairs"]:
                        st.session_state[f"{relationship_type}_pairs"].append((new_from, new_to))
                    else:
                        st.warning("This relationship already exists!")
                else:
                    st.warning("Cannot create self-relationship")
                st.rerun()

    return st.session_state[f"{relationship_type}_pairs"]

def _add_tier_controls(df: pd.DataFrame) -> Dict[int, List[str]]:
    """Create UI for tiered knowledge using PROCESSED column names."""
    tiers = {}
    cols = df.columns.tolist()  # Uses processed column names
    
    st.markdown("**Tier Ordering** (Low tiers cause high tiers, set low tiers for for example sex and age like 0 and outcome variables to higher tiers.)")
    
    # Create a dictionary to track assignments
    assigned_vars = set()
    
    for var in cols:
        with st.container():
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"`{var}`")
            with col2:
                tier = st.number_input(
                    "Tier level",
                    min_value=0,
                    max_value=100,
                    value=5,
                    key=f"tier_{var}",
                    help=f"Temporal tier for {var}. Low tiers cause higher tiers."
                )
                
                if var in assigned_vars:
                    st.warning("Variable already assigned")
                else:
                    if{tier != 0}:
                        if tier not in tiers:
                            tiers[tier] = []
                        tiers[tier].append(var)
                    assigned_vars.add(var)
    return tiers

def get_background_knowledge(df: pd.DataFrame) -> BackgroundKnowledge:
    """Create background knowledge with validation and display constraints."""
    bk = BackgroundKnowledge()
    constraint_list = []  # To track all constraints
    
    # Create nodes using indices (X1, X2, ...) so it matches with R notation for consistency
    nodes = [GraphNode(f"X{i+1}") for i in range(len(df.columns))]
    # Create mapping from column name to node name (X1, X2, ...)
    column_to_node = {col: f"X{i+1}" for i, col in enumerate(df.columns)}
    
    with st.expander("+ Add Background Knowledge"):
        
        # Get constraints in terms of column names
        forbidden_pairs = _add_relationship_controls(df, "forbidden")
        required_pairs = _add_relationship_controls(df, "required")
        tier_map = _add_tier_controls(df)
        
        # Convert constraints to node names (X1, X2, ...)
        try:
            forbidden_pairs_node = [(column_to_node[from_col], column_to_node[to_col]) for (from_col, to_col) in forbidden_pairs]
            required_pairs_node = [(column_to_node[from_col], column_to_node[to_col]) for (from_col, to_col) in required_pairs]
            tier_map_node = OrderedDict()
            for tier_level in sorted(tier_map.keys()):
                tier_vars = [column_to_node[col] for col in tier_map[tier_level]]
                tier_map_node[tier_level] = tier_vars
        except KeyError as e: #safety check
            st.error(f"Invalid variable selection: {str(e)}")
            return bk 
        
        # Build constraint display using column names
        if forbidden_pairs:
            constraint_list.append("**Forbidden Relationships:**")
            constraint_list.extend([f"⛔ {pair[0]} → {pair[1]}" for pair in forbidden_pairs])
            
        if required_pairs:
            constraint_list.append("\n**Required Relationships:**")
            constraint_list.extend([f"✅ {pair[0]} → {pair[1]}" for pair in required_pairs])
            
        if tier_map:
            for tier_level in sorted(tier_map.keys()):
                vars_str = ", ".join(tier_map[tier_level])
                constraint_list.append(f"Tier {tier_level}: {vars_str}")

        # Display all the constraints
        if constraint_list:
            st.divider()
            st.markdown("\n".join(constraint_list))
    
    valid = True
    try:
        _apply_edge_constraints(bk, nodes, forbidden_pairs_node, required_pairs_node)
        _apply_tier_constraints(bk, nodes, tier_map_node)
    except KeyError as e:
        st.error(f"Invalid variable selection: {str(e)}")
        valid = False
    
    return bk if valid else BackgroundKnowledge()

def _apply_edge_constraints(bk: BackgroundKnowledge, nodes: List[GraphNode], 
                          forbidden_pairs: List[Tuple[str, str]], 
                          required_pairs: List[Tuple[str, str]]):
    """Apply edge constraints using node indices (X0, X1, ...)."""
    node_map = {node.get_name(): node for node in nodes}
    
    #this code should never trigger, but its a nice check
    for from_var, to_var in forbidden_pairs:
        if from_var not in node_map:
            raise KeyError(f"Forbidden source variable not found: {from_var}")
        if to_var not in node_map:
            raise KeyError(f"Forbidden target variable not found: {to_var}")
        bk.add_forbidden_by_node(node_map[from_var], node_map[to_var])
    
    for from_var, to_var in required_pairs:
        if from_var not in node_map:
            raise KeyError(f"Required source variable not found: {from_var}")
        if to_var not in node_map:
            raise KeyError(f"Required target variable not found: {to_var}")
        bk.add_required_by_node(node_map[from_var], node_map[to_var])

def _apply_tier_constraints(
    bk: BackgroundKnowledge,
    nodes: List[GraphNode],
    tier_map: Dict[int, List[str]]
):
    """Apply tiered constraints using node indices (X1, X2, ...)."""
    node_map = {node.get_name(): node for node in nodes}


    for tier_level in sorted(tier_map.keys(), reverse=True):
        tier_vars = [node_map[var] for var in tier_map[tier_level]]
        for node in tier_vars:
            #if tier_level != 0:
                bk.add_node_to_tier(node, tier_level)

