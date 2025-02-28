import os
import sys
import json
import asyncio
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
from pathlib import Path

import instructor
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

# Models for structured data
class WorkflowState(Enum):
    PROMPT_CREATION = "prompt_creation"
    METADATA_GENERATION = "metadata_generation"
    METADATA_ADJUSTMENT = "metadata_adjustment"
    CONTENT_GENERATION = "content_generation"
    CONTENT_REFINEMENT = "content_refinement"
    EXPORT = "export"

class Worldview(BaseModel):
    """世界观设定模型"""
    name: str = Field(..., description="世界观的名称")
    description: str = Field(..., description="世界观的详细描述")

class Outline(BaseModel):
    """大纲模型"""
    chapter_number: int = Field(..., description="章节编号")
    title: str = Field(..., description="章节标题")
    summary: str = Field(..., description="章节的摘要")
    key_events: Optional[List[str]] = Field(None, description="章节中包含的关键事件列表")

class Character(BaseModel):
    """角色模型"""
    name: str = Field(..., description="角色的名称")
    description: str = Field(..., description="角色的描述")
    role: str = Field(..., description="角色在故事中的角色（例如：主角、反派）")
    background: Optional[str] = Field(None, description="角色的背景故事")
    motivation: Optional[str] = Field(None, description="角色的动机")
    relationships: Optional[Dict[str, str]] = Field(None, description="角色与其他角色的关系字典")

class NovelMetadata(BaseModel):
    """小说元数据模型"""
    title: str = Field(..., description="小说的标题")
    genre: List[str] = Field(..., description="小说的类型列表")
    worldview: List[Worldview] = Field(..., description="小说的世界观设定")
    characters: List[Character] = Field(..., description="小说中的角色列表")
    outline: List[Outline] = Field(..., description="小说的大纲列表")
    theme: List[str] = Field(..., description="小说的主题列表")
    style: List[str] = Field(..., description="小说的风格列表")
    target_length: Optional[str] = Field("medium", description="小说的目标长度（短篇、中篇、长篇）")

class Chapter(BaseModel):
    """章节模型"""
    number: int = Field(..., description="章节的编号")
    title: str = Field(..., description="章节的标题")
    summary: str = Field(..., description="章节的摘要")
    content: List[str] = Field(default_factory=list, description="章节的内容列表（段落）")

class Novel(BaseModel):
    """小说模型"""
    metadata: NovelMetadata = Field(..., description="小说的元数据")
    chapters: List[Chapter] = Field(default_factory=list, description="小说的章节列表")

class EnhancedPrompt(BaseModel):
    """增强提示模型"""
    prompt: str = Field(..., description="增强的提示内容")

class RefinedParagraph(BaseModel):
    """精炼段落模型"""
    paragraph: str = Field(..., description="精炼后的段落内容")

class NovelGenerator:
    def __init__(self):
        """Initialize the novel generator."""
        # Load environment variables
        load_dotenv()
        
        # Set up console
        self.console = Console()
        
        # Initialize the state
        self.state = WorkflowState.PROMPT_CREATION
        self.novel = None
        self.current_prompt = ""
        self.enhanced_prompt = ""
        self.current_metadata_focus = None
        self.current_chapter = 0
        self.current_paragraph = 0
        
        # Set up API access
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Initialize model with instructor
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name="models/gemini-2.0-flash",
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )
        
        # Initialize prompt session for command completion
        self.commands = WordCompleter([
            '/help', 
            '/exit', 
            '/save', 
            '/load', 
            '/edit', 
            '/expand', 
            '/confirm', 
            '/generate', 
            '/refine', 
            '/export',
            '/next', 
            '/prev', 
            '/view'
        ])
        self.session = PromptSession(completer=self.commands)
        
    async def start(self):
        """Start the novel generation workflow."""
        self.console.print(Panel.fit(
            "[bold cyan]NovelGen[/bold cyan] - Interactive Novel Generation\n"
            "Type [bold green]/help[/bold green] for available commands.",
            title="Welcome",
            border_style="cyan"
        ))
        
        while True:
            try:
                if self.state == WorkflowState.PROMPT_CREATION:
                    await self.handle_prompt_creation()
                elif self.state == WorkflowState.METADATA_GENERATION:
                    await self.handle_metadata_generation()
                elif self.state == WorkflowState.METADATA_ADJUSTMENT:
                    await self.handle_metadata_adjustment()
                elif self.state == WorkflowState.CONTENT_GENERATION:
                    await self.handle_content_generation()
                elif self.state == WorkflowState.CONTENT_REFINEMENT:
                    await self.handle_content_refinement()
                elif self.state == WorkflowState.EXPORT:
                    await self.handle_export()
                    break
            except KeyboardInterrupt:
                if Confirm.ask("Do you want to exit?"):
                    self.console.print("[yellow]Exiting NovelGen...[/yellow]")
                    break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
                if Confirm.ask("Do you want to continue?"):
                    continue
                else:
                    break
    
    async def handle_prompt_creation(self):
        """Handle the prompt creation state."""
        self.console.print(Panel("[bold]Enter your novel concept[/bold] (or type a command):", 
                               title="Prompt Creation", border_style="green"))
        
        user_input = await self.get_user_input()
        
        if user_input.startswith('/'):
            await self.handle_command(user_input)
            return
        
        self.current_prompt = user_input
        self.console.print("\n[bold]Enhancing your prompt...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[green]Optimizing prompt...", total=None)
            self.enhanced_prompt = await self.enhance_prompt(self.current_prompt)
            progress.update(task, completed=True)
        
        self.console.print(Panel(
            f"[bold]Original:[/bold]\n{self.current_prompt}\n\n"
            f"[bold]Enhanced:[/bold]\n{self.enhanced_prompt}",
            title="Prompt Comparison",
            border_style="cyan"
        ))
        
        if Confirm.ask("Do you want to use the enhanced prompt?"):
            self.current_prompt = self.enhanced_prompt
            self.state = WorkflowState.METADATA_GENERATION
        
    async def handle_metadata_generation(self):
        """Generate novel metadata from the prompt."""
        self.console.print("\n[bold]Generating novel metadata...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[green]Creating your novel world...", total=None)
            self.novel = await self.generate_novel_metadata(self.current_prompt)
            progress.update(task, completed=True)
        
        self.display_novel_metadata()
        self.state = WorkflowState.METADATA_ADJUSTMENT
        self.current_metadata_focus = "worldview"
    
    async def handle_metadata_adjustment(self):
        """Handle the metadata adjustment state."""
        if not self.current_metadata_focus:
            self.current_metadata_focus = "worldview"
            
        self.console.print(Panel(
            self.get_current_metadata_section(),
            title=f"Current Focus: {self.current_metadata_focus.title()}",
            border_style="yellow"
        ))
        
        self.console.print("\nCommands: [bold green]/edit[/bold green], [bold green]/expand[/bold green], [bold green]/confirm[/bold green], [bold green]/next[/bold green], [bold green]/prev[/bold green]")
        
        user_input = await self.get_user_input()
        
        if user_input.startswith('/'):
            await self.handle_command(user_input)
        else:
            self.console.print("[yellow]Please use a command to continue.[/yellow]")
    
    async def handle_content_generation(self):
        """Handle the content generation state."""
        if not self.novel.chapters:
            # Create chapters based on outline
            self.console.print("[bold]Creating chapters from outline...[/bold]")
            await self.create_chapters_from_outline()
        
        total_chapters = len(self.novel.chapters)
        current_chapter = self.novel.chapters[self.current_chapter]
        
        self.console.print(Panel(
            f"[bold]Chapter {current_chapter.number}: {current_chapter.title}[/bold]\n\n"
            f"Summary: {current_chapter.summary}",
            title=f"Generating Chapter {current_chapter.number}/{total_chapters}",
            border_style="blue"
        ))
        
        if not current_chapter.content:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(f"[green]Generating Chapter {current_chapter.number}...", total=None)
                await self.generate_chapter_content(current_chapter)
                progress.update(task, completed=True)
            
            self.display_chapter_content(current_chapter)
        else:
            self.display_chapter_content(current_chapter)
        
        self.console.print("\nCommands: [bold green]/next[/bold green], [bold green]/prev[/bold green], [bold green]/refine[/bold green], [bold green]/confirm[/bold green]")
        
        user_input = await self.get_user_input()
        if user_input.startswith('/'):
            await self.handle_command(user_input)
        else:
            self.console.print("[yellow]Please use a command to continue.[/yellow]")
    
    async def handle_content_refinement(self):
        """Handle the content refinement state."""
        current_chapter = self.novel.chapters[self.current_chapter]
        
        if not current_chapter.content:
            self.console.print("[yellow]This chapter has no content to refine yet.[/yellow]")
            self.state = WorkflowState.CONTENT_GENERATION
            return
        
        # Display paragraphs with numbers
        self.console.print(Panel(
            f"[bold]Chapter {current_chapter.number}: {current_chapter.title}[/bold]",
            title="Content Refinement",
            border_style="magenta"
        ))
        
        for i, paragraph in enumerate(current_chapter.content):
            self.console.print(f"[bold]{i+1}.[/bold] {paragraph[:100]}..." if len(paragraph) > 100 else paragraph)
        
        self.console.print("\nEnter paragraph number to refine, or use a command:")
        user_input = await self.get_user_input()
        
        if user_input.startswith('/'):
            await self.handle_command(user_input)
        else:
            try:
                para_num = int(user_input) - 1
                if 0 <= para_num < len(current_chapter.content):
                    await self.refine_paragraph(current_chapter, para_num)
                else:
                    self.console.print("[red]Invalid paragraph number.[/red]")
            except ValueError:
                self.console.print("[yellow]Please enter a valid paragraph number or command.[/yellow]")
    
    async def handle_export(self):
        """Handle the export state."""
        filename = Prompt.ask("Enter filename to save (without extension)", default=f"{self.novel.metadata.title.lower().replace(' ', '_')}")
        path = Path(f"{filename}.md")
        
        # Create markdown export
        markdown_content = self.generate_markdown()
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        self.console.print(f"[green]Novel successfully exported to [bold]{path}[/bold][/green]")
        
        if Confirm.ask("Do you want to exit NovelGen?"):
            return
        else:
            self.state = WorkflowState.CONTENT_REFINEMENT
    
    async def handle_command(self, command):
        """Handle command inputs."""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            self.display_help()
        elif cmd == '/exit':
            if Confirm.ask("Are you sure you want to exit?"):
                sys.exit(0)
        elif cmd == '/save':
            await self.save_novel()
        elif cmd == '/load':
            await self.load_novel()
        elif cmd == '/edit' and self.state == WorkflowState.METADATA_ADJUSTMENT:
            await self.edit_current_metadata()
        elif cmd == '/expand' and self.state == WorkflowState.METADATA_ADJUSTMENT:
            await self.expand_current_metadata()
        elif cmd == '/confirm':
            await self.confirm_current_state()
        elif cmd == '/generate' and self.state == WorkflowState.CONTENT_GENERATION:
            await self.regenerate_current_chapter()
        elif cmd == '/refine':
            if self.state == WorkflowState.CONTENT_GENERATION:
                self.state = WorkflowState.CONTENT_REFINEMENT
            elif self.state == WorkflowState.CONTENT_REFINEMENT:
                self.state = WorkflowState.CONTENT_GENERATION
        elif cmd == '/export':
            self.state = WorkflowState.EXPORT
        elif cmd == '/next':
            await self.go_to_next()
        elif cmd == '/prev':
            await self.go_to_previous()
        elif cmd == '/view':
            if self.state in [WorkflowState.METADATA_ADJUSTMENT, WorkflowState.CONTENT_GENERATION, WorkflowState.CONTENT_REFINEMENT]:
                self.display_novel_metadata()
        else:
            self.console.print("[yellow]Unknown or invalid command for current state.[/yellow]")
    
    async def enhance_prompt(self, prompt):
        """Enhance the user's original prompt."""
        system_message = """
            你是一个专业中文小说助手。
            你的任务是通过增加深度、具体性和创造性元素来增强用户的小说概念。
            在保持原始构想的同时，使其更加生动和详细。
        """
        
        try:
            response = self.client.chat.completions.create(
                response_model=EnhancedPrompt,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [f"增强这个小说概念：{prompt}"]}
                ]
            )
            return response.prompt
        except Exception as e:
            self.console.print(f"[red]Error enhancing prompt: {str(e)}[/red]")
            return prompt
    
    async def generate_novel_metadata(self, prompt):
        """Generate novel metadata using the enhanced prompt."""
        try:
            metadata = self.client.chat.completions.create(
                response_model=NovelMetadata,
                messages=[
                    {"role": "system", "content": "你是一个创造性小说AI。生成详细的小说元数据。"},
                    {"role": "user", "content": f"根据这个概念创建小说元数据：{prompt}"}
                ]
            )
            return Novel(metadata=metadata)
        except Exception as e:
            self.console.print(f"[red]Error generating metadata: {str(e)}[/red]")
            # Create default metadata as fallback
            return Novel(
                metadata=NovelMetadata(
                    title="Untitled Novel",
                    genre=["Fiction"],
                    worldview="A world waiting to be described...",
                    characters=[
                        Character(
                            name="Protagonist",
                            description="Main character",
                            role="Protagonist"
                        )
                    ],
                    outline=["Chapter 1: Beginning"],
                    theme="To be determined",
                    style="Default"
                )
            )
    
    async def create_chapters_from_outline(self):
        """Create chapter structures from the outline."""
        chapters = []
        for outline_item in self.novel.metadata.outline:
            chapters.append(Chapter(
                number=outline_item.chapter_number,
                title=outline_item.title,
                summary=outline_item.summary
            ))
        
        self.novel.chapters = chapters
    
    async def generate_chapter_content(self, chapter):
        """Generate content for a specific chapter."""
        try:
            system_message = (
                f"你正在写小说'{self.novel.metadata.title}'的章节{chapter.number}: {chapter.title}。"
                f"以'{', '.join(self.novel.metadata.style)}'风格写出流畅、生动的散文。"
                f"专注于主题：'{', '.join(self.novel.metadata.theme)}'。"
            )
            
            # Prepare worldview context
            worldview_context = "\n".join([f"{w.name}: {w.description}" for w in self.novel.metadata.worldview])
            
            context = (
                f"小说世界观：\n{worldview_context}\n"
                f"角色：{', '.join([c.name for c in self.novel.metadata.characters])}\n"
                f"章节摘要：{chapter.summary}"
            )
            
            response = self.client.chat.completions.create(
                response_model=Chapter,
                messages=[
                    {"role": "system", "content": [system_message]},
                    {"role": "user", "content": [f"上下文：\n{context}\n\n请将这一章节分段写出，使其引人入胜，并符合该风格。"]}
                ]
            )
            
            # Split the content into paragraphs
            paragraphs = [p.strip() for p in response.content if p.strip()]
            chapter.content = paragraphs
            
        except Exception as e:
            self.console.print(f"[red]Error generating chapter content: {str(e)}[/red]")
            chapter.content = ["[Content generation failed. Please try again.]"]
    
    async def refine_paragraph(self, chapter, paragraph_index):
        """Refine a specific paragraph."""
        original_paragraph = chapter.content[paragraph_index]
        
        self.console.print(Panel(
            f"[bold]Original Paragraph:[/bold]\n{original_paragraph}",
            title=f"Refining Paragraph {paragraph_index+1}",
            border_style="cyan"
        ))
        
        self.console.print("\nRefine options:")
        self.console.print("1. Rewrite (general improvement)")
        self.console.print("2. Expand (add more detail)")
        self.console.print("3. Condense (make more concise)")
        self.console.print("4. Change style (make more descriptive/dramatic/etc.)")
        self.console.print("5. Custom instruction")
        
        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5"])
        
        instruction = ""
        if choice == "1":
            instruction = "重写此段落，以提升流畅度和清晰度。"
        elif choice == "2":
            instruction = "扩展此段落，增加更多感官细节和描述。"
        elif choice == "3":
            instruction = "压缩此段落，使其更精炼，同时保留原意。"
        elif choice == "4":
            style = Prompt.ask("请输入期望的风格", default="更加生动")
            instruction = f"将此段落重写为{style}风格。"
        elif choice == "5":
            instruction = Prompt.ask("Enter custom instruction")
        
        self.console.print("\n[bold]Refining paragraph...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[green]Refining...", total=None)
            
            try:
                system_message = f"你是一名专业的小说编辑。请润色以下内容，来自《{self.novel.metadata.title}》的第 {chapter.number} 章。"
                
                response = self.client.chat.completions.create(
                    response_model=RefinedParagraph,
                    messages=[
                        {"role": "system", "content": [system_message]},
                        {"role": "user", "content": [f"原始段落：\n\n{original_paragraph}\n\n修改指令：{instruction}"]}
                    ]
                )
                
                refined_paragraph = response.paragraph
                progress.update(task, completed=True)
                
                # Show comparison
                self.console.print(Panel(
                    f"[bold]Original:[/bold]\n{original_paragraph}\n\n"
                    f"[bold]Refined:[/bold]\n{refined_paragraph}",
                    title="Paragraph Comparison",
                    border_style="green"
                ))
                
                if Confirm.ask("Apply this change?"):
                    chapter.content[paragraph_index] = refined_paragraph
                    self.console.print("[green]Paragraph updated successfully![/green]")
                
            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"[red]Error refining paragraph: {str(e)}[/red]")
    
    async def confirm_current_state(self):
        """Confirm the current state and move to the next step."""
        if self.state == WorkflowState.METADATA_ADJUSTMENT:
            if Confirm.ask("Are you satisfied with the novel metadata?"):
                self.state = WorkflowState.CONTENT_GENERATION
        elif self.state == WorkflowState.CONTENT_GENERATION:
            if self.current_chapter == len(self.novel.chapters) - 1:
                if Confirm.ask("You've reached the end of the novel. Ready to export?"):
                    self.state = WorkflowState.EXPORT
            else:
                self.current_chapter += 1
    
    async def go_to_next(self):
        """Navigate to the next item in the current context."""
        if self.state == WorkflowState.METADATA_ADJUSTMENT:
            metadata_sections = ["worldview", "characters", "outline", "theme", "style"]
            current_index = metadata_sections.index(self.current_metadata_focus)
            if current_index < len(metadata_sections) - 1:
                self.current_metadata_focus = metadata_sections[current_index + 1]
            else:
                self.console.print("[yellow]You've reached the end of metadata sections.[/yellow]")
        
        elif self.state == WorkflowState.CONTENT_GENERATION:
            if self.current_chapter < len(self.novel.chapters) - 1:
                self.current_chapter += 1
            else:
                self.console.print("[yellow]You've reached the last chapter.[/yellow]")
    
    async def go_to_previous(self):
        """Navigate to the previous item in the current context."""
        if self.state == WorkflowState.METADATA_ADJUSTMENT:
            metadata_sections = ["worldview", "characters", "outline", "theme", "style"]
            current_index = metadata_sections.index(self.current_metadata_focus)
            if current_index > 0:
                self.current_metadata_focus = metadata_sections[current_index - 1]
            else:
                self.console.print("[yellow]You're at the first metadata section.[/yellow]")
        
        elif self.state in [WorkflowState.CONTENT_GENERATION, WorkflowState.CONTENT_REFINEMENT]:
            if self.current_chapter > 0:
                self.current_chapter -= 1
            else:
                self.console.print("[yellow]You're at the first chapter.[/yellow]")
    
    async def edit_current_metadata(self):
        """Edit the current metadata section."""
        if self.current_metadata_focus == "worldview":
            self.display_worldview()
            option = Prompt.ask("Choose an option", choices=["edit", "add", "remove"])
            
            if option == "edit":
                idx = int(Prompt.ask("Enter worldview number to edit", default="1")) - 1
                if 0 <= idx < len(self.novel.metadata.worldview):
                    world = self.novel.metadata.worldview[idx]
                    self.console.print(f"Editing worldview: [bold]{world.name}[/bold]")
                    
                    name = Prompt.ask("Name", default=world.name)
                    description = Prompt.ask("Description", default=world.description)
                    
                    world.name = name
                    world.description = description
                else:
                    self.console.print("[red]Invalid worldview number.[/red]")
                    
            elif option == "add":
                name = Prompt.ask("Name")
                description = Prompt.ask("Description")
                
                self.novel.metadata.worldview.append(Worldview(
                    name=name,
                    description=description
                ))
                
            elif option == "remove":
                idx = int(Prompt.ask("Enter worldview number to remove", default="1")) - 1
                if 0 <= idx < len(self.novel.metadata.worldview):
                    world = self.novel.metadata.worldview.pop(idx)
                    self.console.print(f"Removed worldview: [bold]{world.name}[/bold]")
                else:
                    self.console.print("[red]Invalid worldview number.[/red]")
                
        elif self.current_metadata_focus == "characters":
            self.display_characters()
            option = Prompt.ask("Choose an option", choices=["edit", "add", "remove"])
            
            if option == "edit":
                idx = int(Prompt.ask("Enter character number to edit", default="1")) - 1
                if 0 <= idx < len(self.novel.metadata.characters):
                    char = self.novel.metadata.characters[idx]
                    self.console.print(f"Editing character: [bold]{char.name}[/bold]")
                    
                    name = Prompt.ask("Name", default=char.name)
                    description = Prompt.ask("Description", default=char.description)
                    role = Prompt.ask("Role", default=char.role)
                    background = Prompt.ask("Background", default=char.background or "")
                    
                    char.name = name
                    char.description = description
                    char.role = role
                    char.background = background if background else None
                    
                else:
                    self.console.print("[red]Invalid character number.[/red]")
                    
            elif option == "add":
                name = Prompt.ask("Name")
                description = Prompt.ask("Description")
                role = Prompt.ask("Role")
                background = Prompt.ask("Background (optional)")
                
                self.novel.metadata.characters.append(Character(
                    name=name,
                    description=description,
                    role=role,
                    background=background if background else None
                ))
                
            elif option == "remove":
                idx = int(Prompt.ask("Enter character number to remove", default="1")) - 1
                if 0 <= idx < len(self.novel.metadata.characters):
                    char = self.novel.metadata.characters.pop(idx)
                    self.console.print(f"Removed character: [bold]{char.name}[/bold]")
                else:
                    self.console.print("[red]Invalid character number.[/red]")
        
        elif self.current_metadata_focus == "outline":
            self.display_outline()
            option = Prompt.ask("Choose an option", choices=["edit", "add", "remove"])
            
            if option == "edit":
                idx = int(Prompt.ask("Enter outline item number to edit", default="1")) - 1
                if 0 <= idx < len(self.novel.metadata.outline):
                    outline = self.novel.metadata.outline[idx]
                    self.console.print(f"Editing outline: Chapter {outline.chapter_number}: {outline.title}")
                    
                    title = Prompt.ask("Title", default=outline.title)
                    chapter_number = int(Prompt.ask("Chapter Number", default=str(outline.chapter_number)))
                    summary = Prompt.ask("Summary", default=outline.summary)
                    
                    # Handle key events if they exist
                    key_events = outline.key_events
                    if key_events and Confirm.ask("Edit key events?"):
                        events_str = ", ".join(key_events)
                        new_events = Prompt.ask("Key Events (comma separated)", default=events_str)
                        key_events = [event.strip() for event in new_events.split(",") if event.strip()]
                    
                    outline.title = title
                    outline.chapter_number = chapter_number
                    outline.summary = summary
                    outline.key_events = key_events
                else:
                    self.console.print("[red]Invalid outline number.[/red]")
                    
            elif option == "add":
                chapter_number = int(Prompt.ask("Chapter Number", default=str(len(self.novel.metadata.outline) + 1)))
                title = Prompt.ask("Title")
                summary = Prompt.ask("Summary")
                key_events_str = Prompt.ask("Key Events (comma separated, optional)")
                key_events = [event.strip() for event in key_events_str.split(",") if event.strip()] if key_events_str else None
                
                # Determine position to insert
                position = Prompt.ask("Add at position (number or 'end')", default="end")
                
                new_outline = Outline(
                    chapter_number=chapter_number,
                    title=title,
                    summary=summary,
                    key_events=key_events
                )
                
                if position == "end":
                    self.novel.metadata.outline.append(new_outline)
                else:
                    try:
                        pos = int(position) - 1
                        self.novel.metadata.outline.insert(pos, new_outline)
                    except (ValueError, IndexError):
                        self.console.print("[red]Invalid position.[/red]")
                        self.novel.metadata.outline.append(new_outline)
                    
            elif option == "remove":
                idx = int(Prompt.ask("Enter outline number to remove", default="1")) - 1
                if 0 <= idx < len(self.novel.metadata.outline):
                    outline = self.novel.metadata.outline.pop(idx)
                    self.console.print(f"Removed outline: [bold]Chapter {outline.chapter_number}: {outline.title}[/bold]")
                else:
                    self.console.print("[red]Invalid outline number.[/red]")
        
        elif self.current_metadata_focus == "theme":
            themes = self.novel.metadata.theme
            self.console.print(f"[bold]Current themes:[/bold] {', '.join(themes)}")
            
            option = Prompt.ask("Choose an option", choices=["edit", "add", "remove"])
            
            if option == "edit":
                themes_str = ", ".join(themes)
                new_themes = Prompt.ask("Enter themes (comma separated)", default=themes_str)
                self.novel.metadata.theme = [theme.strip() for theme in new_themes.split(",") if theme.strip()]
            
            elif option == "add":
                new_theme = Prompt.ask("Enter new theme")
                self.novel.metadata.theme.append(new_theme)
            
            elif option == "remove":
                if not themes:
                    self.console.print("[yellow]No themes to remove.[/yellow]")
                    return
                    
                for i, theme in enumerate(themes, 1):
                    self.console.print(f"{i}. {theme}")
                
                idx = int(Prompt.ask("Enter theme number to remove", default="1")) - 1
                if 0 <= idx < len(themes):
                    removed = themes.pop(idx)
                    self.console.print(f"Removed theme: [bold]{removed}[/bold]")
                else:
                    self.console.print("[red]Invalid theme number.[/red]")
                
        elif self.current_metadata_focus == "style":
            styles = self.novel.metadata.style
            self.console.print(f"[bold]Current styles:[/bold] {', '.join(styles)}")
            
            option = Prompt.ask("Choose an option", choices=["edit", "add", "remove"])
            
            if option == "edit":
                styles_str = ", ".join(styles)
                new_styles = Prompt.ask("Enter styles (comma separated)", default=styles_str)
                self.novel.metadata.style = [style.strip() for style in new_styles.split(",") if style.strip()]
            
            elif option == "add":
                new_style = Prompt.ask("Enter new style")
                self.novel.metadata.style.append(new_style)
            
            elif option == "remove":
                if not styles:
                    self.console.print("[yellow]No styles to remove.[/yellow]")
                    return
                    
                for i, style in enumerate(styles, 1):
                    self.console.print(f"{i}. {style}")
                
                idx = int(Prompt.ask("Enter style number to remove", default="1")) - 1
                if 0 <= idx < len(styles):
                    removed = styles.pop(idx)
                    self.console.print(f"Removed style: [bold]{removed}[/bold]")
                else:
                    self.console.print("[red]Invalid style number.[/red]")

    def display_worldview(self):
        """Display worldview information."""
        if not self.novel or not self.novel.metadata.worldview:
            self.console.print("[yellow]No worldview defined.[/yellow]")
            return
        
        self.console.print("[bold]Worldview Settings:[/bold]")
        for i, world in enumerate(self.novel.metadata.worldview, 1):
            self.console.print(f"{i}. [bold cyan]{world.name}[/bold cyan]")
            self.console.print(f"   {world.description}")
            self.console.print("")
    
    def display_characters(self):
        """Display character information."""
        if not self.novel or not self.novel.metadata.characters:
            self.console.print("[yellow]No characters defined.[/yellow]")
            return
            
        self.console.print("[bold]Characters:[/bold]")
        for i, character in enumerate(self.novel.metadata.characters, 1):
            self.console.print(f"{i}. [bold cyan]{character.name}[/bold cyan] - {character.role}")
            self.console.print(f"   Description: {character.description}")
            if character.background:
                self.console.print(f"   Background: {character.background}")
            if character.motivation:
                self.console.print(f"   Motivation: {character.motivation}")
            if character.relationships:
                self.console.print(f"   Relationships:")
                for rel_name, rel_desc in character.relationships.items():
                    self.console.print(f"     - {rel_name}: {rel_desc}")
            self.console.print("")

    def display_outline(self):
        """Display outline information."""
        if not self.novel or not self.novel.metadata.outline:
            self.console.print("[yellow]No outline defined.[/yellow]")
            return
        
        self.console.print("[bold]Story Outline:[/bold]")
        for i, outline in enumerate(self.novel.metadata.outline, 1):
            self.console.print(f"{i}. [bold cyan]Chapter {outline.chapter_number}: {outline.title}[/bold cyan]")
            self.console.print(f"   Summary: {outline.summary}")
            if outline.key_events:
                self.console.print(f"   Key Events: {', '.join(outline.key_events)}")
            self.console.print("")
    
    async def expand_current_metadata(self):
        """Expand the current metadata section with AI assistance."""
        self.console.print(f"[bold]Expanding {self.current_metadata_focus}...[/bold]")
        
        current_value = self.get_current_metadata_section(raw=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[green]Generating expanded content...", total=None)
            
            try:
                system_message = f"你是一名富有创意的小说 AI。请扩展并完善小说《{self.novel.metadata.title}》的 {self.current_metadata_focus}。"
                
                response = self.client.chat.completions.create(
                    response_model=NovelMetadata,
                    messages=[
                        {"role": "system", "content": [system_message]},
                        {"role": "user", "content": [f"当前 {self.current_metadata_focus}：\n\n{current_value}\n\n请扩展并完善此 {self.current_metadata_focus}，增加更多细节和创意。"]}
                    ]
                )
                
                expanded_content = response[f"{self.current_metadata_focus}"].strip()
                progress.update(task, completed=True)
                
                # Show comparison
                self.console.print(Panel(
                    f"[bold]Original:[/bold]\n{current_value}\n\n"
                    f"[bold]Expanded:[/bold]\n{expanded_content}",
                    title=f"Expanded {self.current_metadata_focus.title()}",
                    border_style="green"
                ))
                
                if Confirm.ask("Apply this expanded version?"):
                    self.update_metadata_section(expanded_content)
                    self.console.print("[green]Content updated successfully![/green]")
                
            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"[red]Error expanding content: {str(e)}[/red]")
    
    async def regenerate_current_chapter(self):
        """Regenerate the current chapter."""
        if self.current_chapter < len(self.novel.chapters):
            chapter = self.novel.chapters[self.current_chapter]
            
            if Confirm.ask(f"Regenerate Chapter {chapter.number}: {chapter.title}?"):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task(f"[green]Regenerating Chapter {chapter.number}...", total=None)
                    await self.generate_chapter_content(chapter)
                    progress.update(task, completed=True)
                
                self.display_chapter_content(chapter)
    
    async def get_user_input(self):
        """Get user input with command completion."""
        user_input = await self.session.prompt_async(">>> ")
        return user_input

    async def get_multiline_input(self, prompt_text):
        """Get multiline input from user."""
        self.console.print(f"\n{prompt_text} (Enter an empty line to finish)")
        lines = []
        while True:
            line = await self.get_user_input()
            if not line:
                break
            lines.append(line)
        return "\n".join(lines)

    def get_current_metadata_section(self, raw=False):
        """Get the content of the current metadata section."""
        if not self.novel:
            return "No metadata available."
        
        if self.current_metadata_focus == "worldview":
            if raw:
                return "\n".join([f"{w.name}: {w.description}" for w in self.novel.metadata.worldview])
            else:
                return self.format_worldview()
        elif self.current_metadata_focus == "characters":
            if raw:
                return "\n".join([f"{c.name}: {c.description}" for c in self.novel.metadata.characters])
            else:
                return self.format_characters()
        elif self.current_metadata_focus == "outline":
            if raw:
                return "\n".join(self.novel.metadata.outline)
            else:
                return self.format_outline()
        elif self.current_metadata_focus == "theme":
            return ", ".join(self.novel.metadata.theme)
        elif self.current_metadata_focus == "style":
            return ", ".join(self.novel.metadata.style)
        else:
            return "Unknown metadata section."

    def update_metadata_section(self, new_content):
        """Update the current metadata section with new content."""
        if self.current_metadata_focus == "worldview":
            # Parse the new worldview content
            # This is simplified and might need more robust parsing
            try:
                worldviews = []
                lines = new_content.split("\n")
                for line in lines:
                    if ":" in line:
                        name, description = line.split(":", 1)
                        worldviews.append(Worldview(name=name.strip(), description=description.strip()))
                if worldviews:
                    self.novel.metadata.worldview = worldviews
            except Exception as e:
                self.console.print(f"[red]Error parsing worldview: {str(e)}[/red]")
        elif self.current_metadata_focus == "theme":
            # Convert string to list of themes
            themes = [theme.strip() for theme in new_content.split(",") if theme.strip()]
            self.novel.metadata.theme = themes
        elif self.current_metadata_focus == "style":
            # Convert string to list of styles
            styles = [style.strip() for style in new_content.split(",") if style.strip()]
            self.novel.metadata.style = styles
        elif self.current_metadata_focus == "outline":
            # This is a simplified approach
            try:
                outlines = []
                current_chapter = 1
                lines = [line.strip() for line in new_content.split("\n") if line.strip()]
                
                i = 0
                while i < len(lines):
                    if i+1 < len(lines):
                        title = lines[i]
                        summary = lines[i+1]
                        chapter_num = current_chapter
                        
                        # Try to extract chapter number from title
                        if "Chapter" in title and ":" in title:
                            try:
                                chapter_part = title.split(":", 1)[0].strip()
                                chapter_num = int(chapter_part.replace("Chapter", "").strip())
                                current_chapter = chapter_num
                            except:
                                pass
                        
                        outlines.append(Outline(
                            chapter_number=chapter_num,
                            title=title,
                            summary=summary,
                            key_events=None
                        ))
                        current_chapter += 1
                        i += 2
                    else:
                        # If only one line left, treat it as a title with empty summary
                        outlines.append(Outline(
                            chapter_number=current_chapter,
                            title=lines[i],
                            summary="",
                            key_events=None
                        ))
                        i += 1
                
                if outlines:
                    self.novel.metadata.outline = outlines
            except Exception as e:
                self.console.print(f"[red]Error parsing outline: {str(e)}[/red]")
        elif self.current_metadata_focus == "characters":
            # Character update from text is complex and might need a more structured approach
            self.console.print("[yellow]Character update from text not fully implemented.[/yellow]")

    def display_novel_metadata(self):
        """以更均衡的布局展示完整的小说元数据。"""
        main_layout = Layout(name="root")
        # 将布局分为头部（标题）和主体（详细信息）
        main_layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )

        # 头部：展示小说标题
        main_layout["header"].update(
            Panel(f"[bold cyan]{self.novel.metadata.title}[/bold cyan]",
                title="Novel Title", border_style="bright_blue")
        )

        # 主体部分拆分为左右两栏
        body_layout = Layout()
        body_layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2)
        )

        # 左侧：展示主要元数据（篇幅、类型、主题、风格）
        left_panel = Layout()
        left_panel.split_column(
            Layout(Panel(self.novel.metadata.target_length.title(),
                        title="Length", border_style="magenta"), size=3),
            Layout(Panel(f"{', '.join(self.novel.metadata.genre)}",
                        title="Genre", border_style="red"), size=3),
            Layout(Panel(f"{', '.join(self.novel.metadata.theme)}",
                        title="Theme", border_style="red"), size=3),
            Layout(Panel(f"{', '.join(self.novel.metadata.style)}",
                        title="Style", border_style="red"), size=3)
        )

        # 右侧：展示详细信息（世界观、人物和大纲）
        right_panel = Layout()
        right_panel.split_column(
            Layout(Panel(self.format_worldview(),
                        title="Worldview", border_style="yellow")),
            Layout(Panel(self.format_characters(),
                        title="Characters", border_style="yellow")),
            Layout(Panel(self.format_outline(),
                        title="Outline", border_style="blue"))
        )

        body_layout["left"].update(left_panel)
        body_layout["right"].update(right_panel)
        main_layout["body"].update(body_layout)

        self.console.print(main_layout)

    def format_worldview(self):
        """Format worldview information for display."""
        if not self.novel or not self.novel.metadata.worldview:
            return "No worldview defined."
        
        table = Table(show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        
        for world in self.novel.metadata.worldview:
            table.add_row(world.name, world.description)
        
        return table

    def format_characters(self):
        """Format character information for display."""
        if not self.novel or not self.novel.metadata.characters:
            return "No characters defined."
        
        table = Table(show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Description", style="green")
        
        for char in self.novel.metadata.characters:
            table.add_row(char.name, char.role, char.description)
        
        return table

    def format_outline(self):
        """Format outline for display."""
        if not self.novel or not self.novel.metadata.outline:
            return "No outline defined."
        
        table = Table(show_header=True)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Title", style="green")
        table.add_column("Summary", style="yellow")
        
        for outline in self.novel.metadata.outline:
            table.add_row(
                str(outline.chapter_number), 
                outline.title,
                outline.summary[:50] + "..." if len(outline.summary) > 50 else outline.summary
            )
        
        return table

    def display_chapter_content(self, chapter):
        """Display the content of a chapter."""
        self.console.print(Panel(
            f"[bold]{chapter.title}[/bold]\n\n" +
            "\n\n".join(chapter.content),
            title=f"Chapter {chapter.number}",
            border_style="blue"
        ))

    def display_help(self):
        """Display help information."""
        help_text = {
            "/help": "Show this help",
            "/exit": "Exit the application",
            "/save": "Save the novel to a file",
            "/load": "Load a novel from a file",
            "/edit": "Edit current metadata or content",
            "/expand": "Expand current metadata with AI",
            "/confirm": "Confirm and proceed to next stage",
            "/generate": "Generate or regenerate chapter content",
            "/refine": "Toggle refinement mode for content",
            "/export": "Export the novel to markdown",
            "/next": "Go to next item (chapter, section)",
            "/prev": "Go to previous item",
            "/view": "View the complete novel metadata"
        }
        
        table = Table(title="Available Commands")
        table.add_column("Command", style="green")
        table.add_column("Description", style="cyan")
        
        for cmd, desc in help_text.items():
            table.add_row(cmd, desc)
        
        self.console.print(table)
    
    def generate_markdown(self):
        """Generate markdown output for the novel."""
        # Metadata section
        md = f"---\n"
        md += f"title: {self.novel.metadata.title}\n"
        md += f"genre: {', '.join(self.novel.metadata.genre)}\n"
        md += f"theme: {', '.join(self.novel.metadata.theme)}\n"
        md += f"style: {', '.join(self.novel.metadata.style)}\n"
        md += f"created: {datetime.now().strftime('%Y-%m-%d')}\n"
        
        # Worldview section
        md += "worldview:\n"
        for world in self.novel.metadata.worldview:
            md += f"  - name: {world.name}\n"
            md += f"    description: >\n      {world.description.replace('\n', '\n      ')}\n"
        
        # Characters section
        md += "characters:\n"
        for char in self.novel.metadata.characters:
            md += f"  - name: {char.name}\n"
            md += f"    role: {char.role}\n"
            md += f"    description: {char.description}\n"
            if char.background:
                md += f"    background: {char.background}\n"
        
        md += "---\n\n"
        
        # Title and intro
        md += f"# {self.novel.metadata.title}\n\n"
        
        # Chapters
        for chapter in self.novel.chapters:
            md += f"## Chapter {chapter.number}: {chapter.title}\n\n"
            for paragraph in chapter.content:
                md += f"{paragraph}\n\n"
        
        return md
    
    async def save_novel(self):
        """Save the novel to a JSON file."""
        if not self.novel:
            self.console.print("[yellow]No novel to save.[/yellow]")
            return
        
        filename = Prompt.ask("Enter filename to save", default=f"{self.novel.metadata.title.lower().replace(' ', '_')}")
        path = Path(f"{filename}.json")
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.novel.model_dump_json(indent=2))
            self.console.print(f"[green]Novel saved to {path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving novel: {str(e)}[/red]")

    async def load_novel(self):
        """Load a novel from a JSON file."""
        filename = Prompt.ask("Enter filename to load")
        path = Path(filename)
        
        if not path.exists():
            if not path.suffix:
                path = Path(f"{filename}.json")
            
            if not path.exists():
                self.console.print(f"[red]File {path} not found.[/red]")
                return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                novel_data = json.load(f)
            
            self.novel = Novel.model_validate(novel_data)
            self.console.print(f"[green]Novel loaded from {path}[/green]")
            self.state = WorkflowState.METADATA_ADJUSTMENT
            self.current_metadata_focus = "worldview"
        except Exception as e:
            self.console.print(f"[red]Error loading novel: {str(e)}[/red]")

if __name__ == "__main__":
    """Main entry point for the application."""
    generator = NovelGenerator()
    try:
        asyncio.run(generator.start())
    except KeyboardInterrupt:
        print("\nExiting NovelGen...")