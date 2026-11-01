"""
Command Line Interface for the Natural Language Data Analysis System.
Provides an interactive CLI for data loading, model configuration, and analysis.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from ..app.main import DataAnalysisApp
from ..llm.models import ProviderType

console = Console()

# Global app instance
app = DataAnalysisApp()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Natural Language Data Analysis System CLI"""
    console.print(Panel.fit(
        "[bold blue]Natural Language Data Analysis System[/bold blue]\n"
        "Analyze your data using natural language with configurable LLM backends",
        title="Welcome"
    ))


@cli.command()
@click.option('--provider', type=click.Choice(['openai', 'anthropic', 'local']), 
              help='LLM provider to configure')
@click.option('--model', help='Model name')
@click.option('--api-key', help='API key for the provider')
@click.option('--base-url', help='Base URL for local providers')
def configure_model(provider: Optional[str], model: Optional[str], 
                   api_key: Optional[str], base_url: Optional[str]):
    """Configure LLM models for analysis"""
    
    async def _configure():
        if not provider:
            provider_choice = Prompt.ask(
                "Select provider", 
                choices=['openai', 'anthropic', 'local'],
                default='local'
            )
        else:
            provider_choice = provider
        
        if not model:
            if provider_choice == 'openai':
                model_suggestions = ['gpt-4', 'gpt-3.5-turbo']
            elif provider_choice == 'anthropic':
                model_suggestions = ['claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
            else:
                model_suggestions = ['codellama', 'llama2', 'mistral']
            
            model_choice = Prompt.ask(
                f"Model name for {provider_choice}",
                default=model_suggestions[0]
            )
        else:
            model_choice = model
        
        if provider_choice in ['openai', 'anthropic'] and not api_key:
            api_key_input = Prompt.ask(f"API key for {provider_choice}", password=True)
        else:
            api_key_input = api_key
        
        if provider_choice == 'local' and not base_url:
            base_url_input = Prompt.ask("Base URL for local model", default="http://localhost:11434")
        else:
            base_url_input = base_url
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Configuring model...", total=None)
            
            success = await app.configure_model(
                provider=provider_choice,
                model_name=model_choice,
                api_key=api_key_input,
                base_url=base_url_input
            )
        
        if success:
            console.print(f"[green]✓[/green] Model '{model_choice}' configured successfully")
        else:
            console.print(f"[red]✗[/red] Failed to configure model '{model_choice}'")
    
    asyncio.run(_configure())


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--name', help='Dataset name')
def load_data(file_path: str, name: Optional[str]):
    """Load data from a file (CSV, Excel, JSON)"""
    
    async def _load():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading data...", total=None)
            
            success = await app.load_data(file_path, name)
        
        if success:
            console.print(f"[green]✓[/green] Data loaded successfully from '{file_path}'")
            
            # Show data info
            data_info = app.get_data_info()
            if 'basic_stats' in data_info:
                stats = data_info['basic_stats']
                console.print(f"Dataset shape: {stats['shape']}")
                console.print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
                console.print(f"Columns: {', '.join(data_info['column_names'][:5])}")
                if len(data_info['column_names']) > 5:
                    console.print(f"... and {len(data_info['column_names']) - 5} more")
        else:
            console.print(f"[red]✗[/red] Failed to load data from '{file_path}'")
    
    asyncio.run(_load())


@cli.command()
@click.option('--name', default='sample', help='Sample dataset name')
def create_sample(name: str):
    """Create sample data for demonstration"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating sample data...", total=None)
        
        success = app.create_sample_data(name)
    
    if success:
        console.print(f"[green]✓[/green] Sample dataset '{name}' created successfully")
        
        # Show data info
        data_info = app.get_data_info()
        if 'basic_stats' in data_info:
            stats = data_info['basic_stats']
            console.print(f"Dataset shape: {stats['shape']}")
            console.print(f"Columns: {', '.join(data_info['column_names'])}")
    else:
        console.print(f"[red]✗[/red] Failed to create sample dataset '{name}'")


@cli.command()
def analyze():
    """Start interactive analysis session"""
    
    console.print(Panel(
        "[bold]Interactive Analysis Session[/bold]\n"
        "Enter natural language queries to analyze your data.\n"
        "Type 'quit' or 'exit' to end the session.",
        title="Analysis Mode"
    ))
    
    async def _analyze():
        # Check if system is ready
        if not app.is_initialized:
            console.print("[red]✗[/red] System not initialized. Please load data and configure models first.")
            return
        
        while True:
            try:
                query = Prompt.ask("\n[bold blue]Analysis Query[/bold blue]")
                
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("Goodbye!")
                    break
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Analyzing...", total=None)
                    
                    result = await app.analyze(query)
                
                # Display results
                console.print("\n" + "="*60)
                
                if result['success']:
                    console.print(f"[green]✓[/green] Analysis completed successfully")
                    console.print(f"[bold]Intent:[/bold] {result['parsed_request']['intent']}")
                    console.print(f"[bold]Columns used:[/bold] {', '.join(result['parsed_request']['columns_used'])}")
                    console.print(f"[bold]Model used:[/bold] {result['model_info']['model_used']}")
                    console.print(f"[bold]Execution time:[/bold] {result['execution']['execution_time']:.2f}s")
                    
                    # Show generated code
                    console.print("\n[bold]Generated Code:[/bold]")
                    console.print(Panel(result['generated_code'], title="Python Code"))
                    
                    # Show output
                    if result['execution']['output']:
                        console.print("\n[bold]Output:[/bold]")
                        console.print(result['execution']['output'])
                    
                    # Show plots
                    if result['execution']['plots_generated']:
                        console.print(f"\n[bold]Plots generated:[/bold] {', '.join(result['execution']['plots_generated'])}")
                    
                    # Show cost if available
                    if result['model_info']['cost'] > 0:
                        console.print(f"[bold]Cost:[/bold] ${result['model_info']['cost']:.4f}")
                
                else:
                    console.print(f"[red]✗[/red] Analysis failed: {result['error']}")
                
                console.print("="*60)
                
            except KeyboardInterrupt:
                console.print("\nGoodbye!")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
    
    asyncio.run(_analyze())


@cli.command()
def status():
    """Show system status and information"""
    
    # Data information
    data_info = app.get_data_info()
    
    if 'error' not in data_info:
        console.print(Panel(
            f"[bold]Datasets:[/bold] {len(data_info.get('datasets', []))}\n"
            f"[bold]Active Dataset:[/bold] {data_info.get('active_dataset', 'None')}\n"
            f"[bold]Initialized:[/bold] {'Yes' if app.is_initialized else 'No'}",
            title="Data Status"
        ))
        
        if 'basic_stats' in data_info:
            stats = data_info['basic_stats']
            
            table = Table(title="Dataset Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Shape", f"{stats['shape'][0]} rows × {stats['shape'][1]} columns")
            table.add_row("Memory Usage", f"{stats['memory_usage_mb']:.2f} MB")
            table.add_row("Numeric Columns", str(len(stats['numeric_columns'])))
            table.add_row("Categorical Columns", str(len(stats['categorical_columns'])))
            
            console.print(table)
    else:
        console.print("[red]No data loaded[/red]")
    
    # Model information
    model_info = app.get_model_info()
    
    models_table = Table(title="Configured Models")
    models_table.add_column("Model", style="cyan")
    models_table.add_column("Provider", style="green")
    models_table.add_column("Active", style="yellow")
    models_table.add_column("Success Rate", style="magenta")
    
    for model in model_info.get('available_models', []):
        models_table.add_row(
            model['name'],
            model['provider'],
            "✓" if model['is_active'] else "",
            f"{model['stats']['success_rate']:.1f}%"
        )
    
    console.print(models_table)
    
    # Usage statistics
    usage_stats = model_info.get('usage_stats', {})
    if usage_stats.get('total_requests', 0) > 0:
        console.print(Panel(
            f"[bold]Total Requests:[/bold] {usage_stats['total_requests']}\n"
            f"[bold]Success Rate:[/bold] {usage_stats['overall_success_rate']:.1f}%\n"
            f"[bold]Total Cost:[/bold] ${usage_stats['total_cost']:.4f}\n"
            f"[bold]Average Execution Time:[/bold] {usage_stats['avg_execution_time']:.2f}s",
            title="Usage Statistics"
        ))


@cli.command()
@click.option('--limit', default=5, help='Number of recent sessions to show')
def history(limit: int):
    """Show analysis session history"""
    
    sessions = app.get_session_history(limit)
    
    if not sessions:
        console.print("[yellow]No analysis sessions found[/yellow]")
        return
    
    table = Table(title=f"Recent Analysis Sessions (Last {limit})")
    table.add_column("Session", style="cyan")
    table.add_column("Query", style="white", max_width=40)
    table.add_column("Intent", style="green")
    table.add_column("Success", style="yellow")
    table.add_column("Model", style="magenta")
    table.add_column("Cost", style="red")
    
    for session in sessions:
        table.add_row(
            session['session_id'][-8:],  # Show last 8 chars
            session['user_request'][:37] + "..." if len(session['user_request']) > 40 else session['user_request'],
            session['intent'],
            "✓" if session['success'] else "✗",
            session['model_used'],
            f"${session['cost']:.4f}" if session['cost'] > 0 else "Free"
        )
    
    console.print(table)


@cli.command()
@click.option('--output', '-o', help='Output file for export')
def export_data(output: Optional[str]):
    """Export the current dataset"""
    
    if not app.data_manager.get_active_dataframe() is not None:
        console.print("[red]✗[/red] No data loaded to export")
        return
    
    if not output:
        output = Prompt.ask("Output file path", default="export.csv")
    
    try:
        if output.lower().endswith('.csv'):
            success = app.data_manager.export_csv(file_path=output)
        elif output.lower().endswith(('.xlsx', '.xls')):
            success = app.data_manager.export_excel(file_path=output)
        else:
            console.print("[red]✗[/red] Unsupported export format. Use .csv or .xlsx")
            return
        
        if success:
            console.print(f"[green]✓[/green] Data exported to '{output}'")
        else:
            console.print(f"[red]✗[/red] Failed to export data to '{output}'")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Export failed: {str(e)}")


if __name__ == "__main__":
    cli() 