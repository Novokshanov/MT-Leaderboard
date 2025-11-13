import gradio as gr
import pandas as pd
import os
from glob import glob

lang_dict = {
    'eng_Latn': '',
    'fra_Latn': '',
    'cmn_Hans': 'zho_Hans', # zho_Hans in nllb project and original dataset
    'deu_Latn': '',
    'ita_Latn': '',
    'spa_Latn': '',
    'por_Latn': 'por_Latn_braz1246',
    'bel_Cyrl': '',
    'ukr_Cyrl': '',
    'kir_Cyrl': '',
    'uzn_Latn': '',
    'tgk_Cyrl': '',
    'azj_Latn': '',
    'hye_Armn': '',
    'kaz_Cyrl': '',
    'tuk_Latn': '',
    'khk_Cyrl': '',
    'tur_Latn': '',
    'arb_Arab': 'arz_Arab',
    'pes_Arab': '',
    'hin_Deva': '',
    'heb_Hebr': '',
    'prs_Arab': '',
    'pbt_Arab': '',
    'jpn_Jpan': '',
    'kor_Hang': 'kor_Kore',
    'tha_Thai': '',
    'vie_Latn': '',
    'ind_Latn': '',
    'bak_Cyrl': '',
    'chv_Cyrl': '',
    'myv_Cyrl': '',
    'tat_Cyrl': '',
    'ydd_Hebr': '',
    'rus_Cyrl': ''
}


def normalize_language_codes_in_directory(directory_path, lang_dict):
        # Mapping alternative codes -> target codes
        reverse_lang_map = {}
        for target, alternative in lang_dict.items():
            if alternative:  # If there's an alternative code
                reverse_lang_map[alternative] = target
            else:  # If the target code is the standard one
                reverse_lang_map[target] = target
        
        csv_files = glob(os.path.join(directory_path, "*.csv"))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            if 'Direction' not in df.columns:
                print(f"Wrong file structure in {csv_file}")
                continue
            
            def normalize_direction(direction):
                if '_2_' in direction:
                    source, target = direction.split('_2_', 1)
                    
                    normalized_source = reverse_lang_map.get(source, source)
                    normalized_target = reverse_lang_map.get(target, target)
                    
                    return f"{normalized_source}_2_{normalized_target}"
                else:
                    return direction
                
            def normalize_score(score):
                if score < 1:
                    return round(score * 100, 2)
                else:
                    return round(score, 2)
            
            def round_score(score):
                return round(score, 2)
            
            df['Direction'] = df['Direction'].apply(normalize_direction)
            df['BLEU'] = df['BLEU'].apply(normalize_score)
            df['chrF'] = df['chrF'].apply(round_score)
            df['chrF++'] = df['chrF++'].apply(round_score)
            df['Meteor'] = df['Meteor'].apply(normalize_score)
            df['Comet-wmt22'] = df['Comet-wmt22'].apply(normalize_score)
            df['XComet-XXL'] = df['XComet-XXL'].apply(normalize_score)
            
            df.to_csv(csv_file, index=False)


def get_available_metrics(benchmark_dir):
        if not benchmark_dir or not os.path.exists(benchmark_dir):
            print(f"Benchmark directory does not exist: {benchmark_dir}")
            return []
        
        csv_files = glob(os.path.join(benchmark_dir, "*.csv"))
        print(f"CSV files in {benchmark_dir}: {csv_files}")
        
        if not csv_files:
            print(f"No CSV files found in {benchmark_dir}")
            return []
        
        try:
            df = pd.read_csv(csv_files[0])
            print(f"Columns in {csv_files[0]}: {list(df.columns)}")
            return [col for col in df.columns if col != 'Direction']
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []


def get_available_directions(benchmark_dir):
        if not benchmark_dir or not os.path.exists(benchmark_dir):
            return []
        
        csv_files = glob(os.path.join(benchmark_dir, "*.csv"))
        if not csv_files:
            return []
        
        try:
            df = pd.read_csv(csv_files[0])
            return sorted(df['Direction'].tolist())
        except Exception as e:
            print(f"Error reading CSV file for directions: {e}")
            return []


def process_model_name(original_name):
        """Process the original filename to extract and format the model name"""
        try:
            # Split by "-" and take the second to last element
            parts = original_name.split("-")
            if len(parts) >= 2:
                model_name = parts[-2]
            else:
                model_name = original_name  # fallback if there's no "-" or only one part
        except:
            model_name = original_name  # fallback if splitting fails
        
        return model_name


def get_available_models(benchmark_dir):
        if not benchmark_dir or not os.path.exists(benchmark_dir):
            return []
        
        csv_files = glob(os.path.join(benchmark_dir, "*.csv"))
        if not csv_files:
            return []
        
        try:
            models = []
            for csv_file in csv_files:
                original_name = os.path.basename(csv_file).replace('.csv', '')
                processed_name = process_model_name(original_name)
                models.append(processed_name)
            return sorted(models)
        except Exception as e:
            print(f"Error reading CSV files for models: {e}")
            return []

def build_app():
    # Find all benchmark directories
    benchmark_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and '-scores' in d]
    print(f"Found benchmark directories: {benchmark_dirs}")

    for benchmark_dir in benchmark_dirs:
        print(f"Normalizing language codes in {benchmark_dir}...")
        normalize_language_codes_in_directory(benchmark_dir, lang_dict)
        print(f"Completed normalization in {benchmark_dir}")
    
    # Define default score zones for each metric
    default_zones = {
        'BLEU': {
            'low': {'threshold': 15, 'color': '#ffc0c0'},  # pastel red
            'medium': {'threshold': 25, 'color': '#ffe0b3'},  # pastel orange
            'high': {'color': '#d0f0c0'}  # pastel green
        },
        'chrF': {
            'low': {'threshold': 35.0, 'color': '#ffc0c0'},  # pastel red
            'medium': {'threshold': 50.0, 'color': '#ffe0b3'},  # pastel orange
            'high': {'color': '#d0f0c0'}  # pastel green
        },
        'chrF++': {
            'low': {'threshold': 35.0, 'color': '#ffc0c0'},  # pastel red
            'medium': {'threshold': 50.0, 'color': '#ffe0b3'},  # pastel orange
            'high': {'color': '#d0f0c0'}  # pastel green
        },
        'Meteor': {
            'low': {'threshold': 40.0, 'color': '#ffc0c0'},  # pastel red
            'medium': {'threshold': 55.0, 'color': '#ffe0b3'},  # pastel orange
            'high': {'color': '#d0f0c0'}  # pastel green
        },
        'Comet-wmt22': {
            'low': {'threshold': 75.0, 'color': '#ffc0c0'},  # pastel red
            'medium': {'threshold': 87.0, 'color': '#ffe0b3'},  # pastel orange
            'high': {'color': '#d0f0c0'}  # pastel green
        },
        'XComet-XXL': {
            'low': {'threshold': 80.0, 'color': '#ffc0c0'},  # pastel red
            'medium': {'threshold': 90.0, 'color': '#ffe0b3'},  # pastel orange
            'high': {'color': '#d0f0c0'}  # pastel green
        }
    }

    def create_results_table(benchmark_dir, metric, selected_directions, selected_models, transpose):
        if not benchmark_dir or not metric:
            return pd.DataFrame()
        
        csv_files = glob(os.path.join(benchmark_dir, "*.csv"))
        if not csv_files:
            return pd.DataFrame()
        
        results = {}
        all_directions = set()
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                original_model_name = os.path.basename(csv_file).replace('.csv', '')
                processed_model_name = process_model_name(original_model_name)
                
                # Filter CSV files based on selected models if specified
                if selected_models and processed_model_name not in selected_models:
                    continue
                
                # Filter by selected directions if specified
                if selected_directions:
                    df = df[df['Direction'].isin(selected_directions)]
                    all_directions.update(selected_directions)
                else:
                    all_directions.update(df['Direction'].tolist())
                
                results[processed_model_name] = dict(zip(df['Direction'], df[metric]))
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Use all directions that were selected/found
        direction_list = selected_directions if selected_directions else sorted(all_directions)
        
        # Create the base DataFrame
        result_df = pd.DataFrame(results, index=direction_list)
        
        if not transpose:
            # Transpose: directions become columns, models become rows
            result_df = result_df.T  # Transpose the data
            result_df = result_df.reset_index()
            result_df = result_df.rename(columns={'index': 'Model'})
            # Reorder to ensure Model is the first column
            cols = ['Model'] + [col for col in result_df.columns if col != 'Model']
            result_df = result_df[cols]
        else:
            # Normal: directions become rows, models become columns
            result_df = result_df.reset_index()
            result_df = result_df.rename(columns={'index': 'Direction'})
            # Reorder to ensure Direction is the first column
            cols = ['Direction'] + [col for col in result_df.columns if col != 'Direction']
            result_df = result_df[cols]
        
        return result_df

    # Define CSS for fixed first column with enhanced styling and colorization
    css = """
    #results_table_container {
        overflow-x: auto;
        max-width: 100%;
        margin: 0 auto;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    
    #results_table_container table {
        border-collapse: collapse;
        width: auto;
        min-width: 100%;
        table-layout: auto;
    }
    
    #results_table_container table th,
    #results_table_container table td {
        padding: 10px 15px;
        text-align: left;
        white-space: nowrap;
        border: 1px solid #ddd;
        vertical-align: top;
    }
    
    #results_table_container table th:first-child,
    #results_table_container table td:first-child {
        position: sticky;
        left: 0;
        background-color: #f3f3f3 !important;
        z-index: 100;
        border-right: 3px solid #bbb;
        min-width: 180px;
        font-weight: bold;
    }
    
    #results_table_container table th:first-child {
        background-color: #e9e9e9 !important;
        position: sticky;
        top: 0;
        z-index: 101;
        border-bottom: 2px solid #bbb;
    }
    
    /* Ensure the container has proper overflow handling */
    #results_table {
        overflow: visible !important;
    }
    
    /* Optional: Add subtle styling for better readability */
    #results_table_container table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    #results_table_container table tr:hover {
        background-color: #f5f5f5;
    }
    """

    with gr.Blocks(css=css) as app:
        gr.Markdown("## Machine Translation Benchmark Results")
        
        with gr.Row():
            default_benchmark = benchmark_dirs[0] if benchmark_dirs else ""
            benchmark_dropdown = gr.Dropdown(
                choices=benchmark_dirs,
                label="Select Benchmark Dataset",
                value=default_benchmark,
                interactive=True
            )
        
        with gr.Row():
            metric_dropdown = gr.Dropdown(
                choices=[],
                label="Select Automatic Metric",
                interactive=True
            )
        
        with gr.Row():
            direction_multiselect = gr.Dropdown(
                choices=[],
                label="Select Translation Directions (Leave empty for all)",
                multiselect=True,
                interactive=True
            )
        
        with gr.Row():
            model_multiselect = gr.Dropdown(
                choices=[],
                label="Select Models (Leave empty for all)",
                multiselect=True,
                interactive=True
            )
        
        with gr.Row():
            transpose_checkbox = gr.Checkbox(
                label="Transpose Table (Directions as Columns, Models as Rows)",
                value=False,
                interactive=True
            )
        
        # Add sorting controls
        with gr.Row():
            sort_by = gr.Dropdown(
                choices=["None"],
                label="Sort By Column",
                value="None",
                interactive=True
            )
            sort_order = gr.Radio(
                choices=["Ascending", "Descending"],
                label="Sort Order",
                value="Descending",
                interactive=True
            )

            colorize_button = gr.Button(
                value="Colorize Scores",
                # scale=0.5
                # size='lg'
            )

        with gr.Row():
            with gr.Column(scale=1):
                # Using HTML component to wrap the table with custom container
                results_html = gr.HTML(label="Results Table", elem_id="results_table")
        
        # State to track colorization status
        colorize_state = gr.State(value=False)
        
        def update_metrics(benchmark_dir):
            metrics = get_available_metrics(benchmark_dir)
            print(f"Available metrics: {metrics}")
            return gr.Dropdown(choices=metrics, value=None)
        
        def update_directions(benchmark_dir):
            directions = get_available_directions(benchmark_dir)
            print(f"Available directions: {directions}")
            return gr.Dropdown(choices=directions)
        
        def update_models(benchmark_dir):
            models = get_available_models(benchmark_dir)
            print(f"Available models: {models}")
            return gr.Dropdown(choices=models)
        
        def update_all_dropdowns(benchmark_dir):
            # Update all dropdowns when a benchmark is selected
            metrics = get_available_metrics(benchmark_dir)
            directions = get_available_directions(benchmark_dir)
            models = get_available_models(benchmark_dir)
            
            return [
                gr.Dropdown(choices=metrics, value=None),
                gr.Dropdown(choices=directions),
                gr.Dropdown(choices=models)
            ]
        
        def update_sort_options(benchmark_dir, metric, selected_directions, selected_models, transpose):
            if not benchmark_dir or not os.path.exists(benchmark_dir) or not metric:
                return gr.Dropdown(choices=["None"])
            
            # Create the current table (without statistics for this function)
            current_table = create_results_table(benchmark_dir, metric, selected_directions, selected_models, transpose)
            
            if current_table.empty:
                return gr.Dropdown(choices=["None"])
            
            # Get all columns from the current table (excluding the first identifier column)
            identifier_col = current_table.columns[0]  # This will be 'Direction' or 'Model'
            other_columns = [col for col in current_table.columns if col != identifier_col]
            # Add the statistics columns that will be available after calculation
            all_available_columns = other_columns + ['Mean', 'Std']
            sort_options = ["None"] + all_available_columns  # Include all metric columns and statistics
            
            return gr.Dropdown(choices=sort_options, value="None")
        
        def get_score_color(value, column_name, selected_metric):
            """Get background color for a score based on zones"""
            try:
                value = float(value)
            except (ValueError, TypeError):
                return ''  # Return empty if not a number
            
            # Determine which metric zones to use
            if selected_metric in default_zones:
                zones = default_zones[selected_metric]
                if value < zones['low']['threshold']:
                    return zones['low']['color']
                elif value < zones['medium']['threshold']:
                    return zones['medium']['color']
                else:
                    return zones['high']['color']
            else:
                return ''  # No color if metric not in zones
        
        def create_results_table_with_stats_and_sort(benchmark_dir, metric, selected_directions, selected_models, transpose, sort_by_col, sort_order_val, colorize=False):
            df = create_results_table(benchmark_dir, metric, selected_directions, selected_models, transpose)
            
            if df.empty:
                return df
            
            # Get the identifier column ('Direction' or 'Model')
            identifier_col = df.columns[0]
            # Get all metric columns (everything except the identifier)
            metric_cols = [col for col in df.columns if col != identifier_col]
            
            # Calculate mean and std for each row (across metric columns)
            # Convert to numeric first to handle any string values
            df_for_stats = df[metric_cols].copy()
            df_for_stats = df_for_stats.apply(pd.to_numeric, errors='coerce')
            
            # Calculate mean and std for each row
            df['Mean'] = df_for_stats.mean(axis=1, skipna=True).round(2)
            df['Std'] = df_for_stats.std(axis=1, skipna=True).round(2)
            
            # Apply sorting if a column is specified and it exists in the dataframe
            if sort_by_col and sort_by_col != "None" and sort_by_col in df.columns:
                ascending = sort_order_val == "Ascending"
                try:
                    # Try to convert to numeric for proper sorting
                    df[sort_by_col] = pd.to_numeric(df[sort_by_col], errors='ignore')
                    df = df.sort_values(by=sort_by_col, ascending=ascending)
                except:
                    # If numeric conversion fails, sort as string
                    df = df.sort_values(by=sort_by_col, ascending=ascending)
            
            # Add colorization if requested
            if colorize:
                # Create a styled version of the dataframe
                styled_df = df.copy()
                # Colorize all metric columns and Mean, but not Std
                for col in metric_cols + ['Mean']:  
                    if col in styled_df.columns:
                        styled_df[col] = styled_df[col].apply(
                            lambda x: f'<span style="background-color: {get_score_color(x, col, metric)}; padding: 2px 4px; border-radius: 3px;">{x}</span>' 
                            if pd.notna(x) and get_score_color(x, col, metric) else str(x)
                        )
                return styled_df
            
            return df

        def update_table(benchmark_dir, metric, selected_directions, selected_models, transpose, sort_by_col, sort_order_val, colorize):
            df = create_results_table_with_stats_and_sort(benchmark_dir, metric, selected_directions, selected_models, transpose, sort_by_col, sort_order_val, colorize)
            if df.empty:
                return "<div style='padding: 20px; text-align: center; color: #666;'>No data available</div>"
            
            # Convert dataframe to HTML with custom container
            table_html = df.to_html(escape=False, index=False, table_id="results_table_inner", classes="results-table")
            # Wrap in container div with id
            full_html = f"""
            <div id="results_table_container">
                {table_html}
            </div>
            """
            return full_html
        
        benchmark_dropdown.change(
            fn=update_all_dropdowns,
            inputs=benchmark_dropdown,
            outputs=[metric_dropdown, direction_multiselect, model_multiselect]
        )

        def update_sort_options_wrapper(benchmark_dir, metric, directions, models, transpose):
            return update_sort_options(benchmark_dir, metric, directions, models, transpose)
        
        # Update sort options when any relevant input changes
        metric_dropdown.change(
            fn=update_sort_options_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox],
            outputs=sort_by
        )
        
        direction_multiselect.change(
            fn=update_sort_options_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox],
            outputs=sort_by
        )
        
        model_multiselect.change(
            fn=update_sort_options_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox],
            outputs=sort_by
        )
        
        transpose_checkbox.change(
            fn=update_sort_options_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox],
            outputs=sort_by
        )
        
        # Update table when any of the relevant inputs change - default is not colorized
        def update_table_wrapper(benchmark_dir, metric, directions, models, transpose, sort_by_col, sort_order_val, colorize_val):
            return update_table(benchmark_dir, metric, directions, models, transpose, sort_by_col, sort_order_val, colorize_val)
        
        metric_dropdown.change(
            fn=update_table_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox, sort_by, sort_order, colorize_state],
            outputs=results_html
        )
        
        direction_multiselect.change(
            fn=update_table_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox, sort_by, sort_order, colorize_state],
            outputs=results_html
        )
        
        model_multiselect.change(
            fn=update_table_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox, sort_by, sort_order, colorize_state],
            outputs=results_html
        )
        
        transpose_checkbox.change(
            fn=update_table_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox, sort_by, sort_order, colorize_state],
            outputs=results_html
        )
        
        sort_by.change(
            fn=update_table_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox, sort_by, sort_order, colorize_state],
            outputs=results_html
        )
        
        sort_order.change(
            fn=update_table_wrapper,
            inputs=[benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox, sort_by, sort_order, colorize_state],
            outputs=results_html
        )
        
        def toggle_colorize(colorize_btn_text, colorize_val, benchmark_dir, metric, directions, models, transpose, sort_by_col, sort_order_val):
            current_colorize = colorize_val  # Get current state from colorize_state
            new_text = "Colorize Scores" if current_colorize else "Remove Colorization"
            new_colorize_state = not current_colorize
            return new_text, new_colorize_state, update_table(benchmark_dir, metric, directions, models, transpose, sort_by_col, sort_order_val, new_colorize_state)
        
        colorize_button.click(
            fn=toggle_colorize,
            inputs=[colorize_button, colorize_state, benchmark_dropdown, metric_dropdown, direction_multiselect, model_multiselect, transpose_checkbox, sort_by, sort_order],
            outputs=[colorize_button, colorize_state, results_html]
        )

        if benchmark_dirs:
            app.load(
                fn=update_all_dropdowns,
                inputs=benchmark_dropdown,
                outputs=[metric_dropdown, direction_multiselect, model_multiselect]
            )
    
    return app

if __name__ == "__main__":
    app = build_app()
    app.launch()