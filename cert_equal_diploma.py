import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from itertools import combinations
from collections import Counter

# Assuming 'df' is already pre-cleaned and contains the columns: ID, PROGRAMS, Actual Title

def generate_combinations(df):
    grouped = df.groupby('ID')[['PROGRAMS', 'Actual Title']].apply(lambda x: list(zip(x['PROGRAMS'], x['Actual Title'])))
    grouped = grouped[grouped.apply(len) >= 2]

    combo_counter = Counter()
    combo_meta = {}

    for combos in grouped:
        for combo in combinations(sorted(combos), 2):
            codes = [p[0] for p in combo]
            if combo not in combo_counter:
                combo_meta[combo] = codes
            combo_counter[combo] += 1

    two_associates, one_associate, no_associate = [], [], []

    for combo, count in combo_counter.items():
        starts = [code[0] for code in combo_meta[combo]]
        if starts[0] == 'A' and starts[1] == 'A':
            two_associates.append((combo, count))
        elif 'A' in starts:
            one_associate.append((combo, count))
        else:
            no_associate.append((combo, count))

    def build_combo_df(combo_list):
        data = {
            'Program Combinations': [f"{p1[0]} | {p1[1]}\n\n\n{p2[0]} | {p2[1]}" for (p1, p2), _ in combo_list],
            'Student Count': [count for _, count in combo_list],
            'Program Code Only': [f"{p1[0]} + {p2[0]}" for (p1, p2), _ in combo_list]  # Added column
        }
        return pd.DataFrame(data)

    return (
        build_combo_df(two_associates),
        build_combo_df(one_associate),
        build_combo_df(no_associate)
    )

def display_tab(df, title, top_n, min_count):
    filtered_df = df.copy()
    if filter_mode == 'Top N Combinations':
        filtered_df = filtered_df.nlargest(top_n, 'Student Count')
    else:
        filtered_df = filtered_df[filtered_df['Student Count'] >= min_count]

    st.subheader(title)
    
    # Initialize session state for this tab if not already done
    tab_key = f"selected_{title.replace(' ', '_')}"
    if tab_key not in st.session_state:
        st.session_state[tab_key] = []

    # Create a container for the data display
    col1, col2 = st.columns([1, 2])
    
    # Display data with native Streamlit checkboxes
    with col1:
        #st.write("Program Combinations")
        
        # Dictionary to store checkbox states
        checkbox_states = {}
        selected_programs = []
        
        # Create a checkbox for each row
        for i, row in filtered_df.iterrows():
            program = row['Program Combinations']
            program_key = f"{tab_key}_{i}"
            
            # Use session state to maintain checkbox state
            if program_key not in st.session_state:
                st.session_state[program_key] = program in st.session_state[tab_key]
            
            # Create the checkbox
            checkbox_states[program] = st.checkbox(
                f"{program}\n\n({row['Student Count']} students)",
                key=program_key
            )
            
            # Track selected programs
            if checkbox_states[program]:
                selected_programs.append(program)
        
        # Update session state with current selections
        st.session_state[tab_key] = selected_programs
    
    # Display bar chart
    with col2:
        # Create bar chart using go.Figure for more control
        fig = go.Figure()
        
        # Get the data for the chart
        x_data = filtered_df['Program Code Only'].tolist()
        y_data = filtered_df['Student Count'].tolist()
        
        # Create a color list based on selection
        colors = []
        for prog in filtered_df['Program Combinations']:
            if prog in selected_programs:
                colors.append('darkorange')
            else:
                colors.append('#1f77b4')
        
        # Add the bar trace
        fig.add_trace(go.Bar(
            x=x_data,
            y=y_data,
            marker_color=colors,
            text=y_data,
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{title} - Bar Chart",
            xaxis_tickangle=-30,
            height=500
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Debug information
    with st.expander("Debug Information"):
        st.write(f"Selected Programs: {selected_programs}")
        st.write(f"Number of Selected Items: {len(selected_programs)}")

# --- Streamlit App ---
st.set_page_config(page_title="Top Program Combinations", layout="wide")
st.title("Top Program Combinations Analysis")

# File uploader
uploaded_file = st.file_uploader("\U0001F4C1 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    import io

    file_bytes = uploaded_file.getvalue()

    # Check if it's a ZIP-like file (XLSX)
    if file_bytes[:2] == b'PK':
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Excel file upload failed: {e}")
            st.stop()
    else:
        try:
            decoded = file_bytes.decode('latin1', errors='ignore')
            df = pd.read_csv(io.StringIO(decoded), on_bad_lines='skip')
        except Exception as e:
            st.error(f"CSV file upload failed: {e}")
            st.stop()

    st.success("File successfully loaded!")

    def manage_df(df):
        # drop unneeded columns
        columns_to_drop = ['Current Status', 'Status Date', 'Current End Date', 'Advisor', 'Primary E-Mail', 
                          'Smv Vetben Benefit ', 'Smv Vetben End Date ']
        # Filter to only include columns that exist in the DataFrame
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        # convert IDs to string
        if 'ID' in df.columns:
            df['ID'] = df['ID'].apply(lambda x: str(int(float(x))) if pd.notna(x) else '')

            # drop rows where ID or program is blank
            df = df.dropna(subset=['ID'])
            if 'PROGRAMS' in df.columns:
                df = df.dropna(subset=['PROGRAMS'])

                # find duplicated IDs
                repeated_ids = df['ID'][df['ID'].duplicated(keep=False)]

                # filter original DF to only rows with repeated IDs
                df = df[df['ID'].isin(repeated_ids)]

        return df

    df = manage_df(df)

    two_df, one_df, no_df = generate_combinations(df)
    all_df = pd.concat([two_df, one_df, no_df], ignore_index=True)

    filter_mode = st.radio("Select Filter Mode:", ["Top N Combinations", "Minimum Student Count"])

    top_n = st.slider("Top N", min_value=5, max_value=50, value=10)
    min_count = st.slider("Minimum Student Count", min_value=1, max_value=20, value=5)

    tab1, tab2, tab3, tab4 = st.tabs([
        "All Combinations",
        "Two Associates Degrees",
        "One Associate Degree",
        "Certs/Diploma Only"
    ])

    with tab1:
        display_tab(all_df, "All Program Combinations", top_n, min_count)
    with tab2:
        display_tab(two_df, "Two Associates Degrees", top_n, min_count)
    with tab3:
        display_tab(one_df, "One Associate Degree", top_n, min_count)
    with tab4:
        display_tab(no_df, "Certificates/Diploma Only", top_n, min_count)