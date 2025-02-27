from dotenv import load_dotenv
import instructor
import google.generativeai as genai
import os
import json
from typing import List
from pydantic import BaseModel, Field
import re
from datetime import datetime
import tqdm
import time
import yaml

# 定义小说结构模型
class Character(BaseModel):
    name: str = Field(..., description="角色名称")
    description: str = Field(..., description="角色描述")
    background: str = Field(..., description="角色背景故事")
    personality: str = Field(..., description="角色性格特点")

class Scene(BaseModel):
    title: str = Field(..., description="场景标题")
    description: str = Field(..., description="场景描述")
    time: str = Field(..., description="场景发生的时间")
    location: str = Field(..., description="场景发生的地点")

class Chapter(BaseModel):
    title: str = Field(..., description="章节标题")
    scenes: List[Scene] = Field(..., description="章节包含的场景")
    content: str = Field(..., description="章节内容，应当包含足够的细节和描写，每章节约2000-3000字")

class WorldSetting(BaseModel):
    name: str = Field(..., description="世界观名称")
    description: str = Field(..., description="世界观描述")
    rules: List[str] = Field(..., description="世界规则列表")
    history: str = Field(..., description="世界历史背景")

class Novel(BaseModel):
    title: str = Field(..., description="小说标题")
    author: str = Field(..., description="作者名称")
    genre: List[str] = Field(..., description="小说类型标签")
    summary: str = Field(..., description="小说摘要")
    world_setting: WorldSetting = Field(..., description="世界观设定")
    characters: List[Character] = Field(..., description="小说角色列表")
    chapters: List[Chapter] = Field(..., description="小说章节")
    tags: List[str] = Field(default=[], description="小说相关标签")

class PolishedContent(BaseModel):
    content: str = Field(..., description="润色后的正文内容，不包含任何非故事内容的前缀或后缀")

class ExtractedTags(BaseModel):
    tags: List[str] = Field(..., description="从小说中提取的标签列表")

class NovelGenerator:
    def __init__(self, prompts):
        """初始化小说生成器"""
        # Load environment variables
        load_dotenv() 

        # 设置API密钥
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # 初始化模型
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name="models/gemini-2.0-flash",
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )
        
        # 设置提示词
        self.prompts = prompts
        
    def generate_novel(self, prompt: str) -> Novel:
        """根据提示生成完整小说"""
        print("正在生成小说结构...")
        
        # 创建进度条
        progress_bar = tqdm.tqdm(total=100, desc="正在生成小说", unit="%")
        
        # 更新进度条以模拟进度
        progress_bar.update(10)
        time.sleep(1)
        
        novel = self.client.chat.completions.create(
            response_model=Novel,
            messages=[
                {"role": "system", "content": self.prompts['generate_novel']},
                {"role": "user", "content": f"请根据以下提示创作一篇中短篇小说: {prompt}"}
            ]
        )
        
        # 完成进度条
        progress_bar.update(90)
        progress_bar.close()
        
        print(f"小说《{novel.title}》生成完成！")
        return novel
    
    def polish_content(self, content: str, style_prompt: str) -> str:
        """使用AI润色内容"""
        print(f"正在根据风格'{style_prompt}'润色内容...")
        
        # 创建进度条
        progress_bar = tqdm.tqdm(total=100, desc="正在润色内容", unit="%")
        progress_bar.update(20)
        
        # 使用instructor代替直接调用模型，确保输出格式符合预期
        polished = self.client.chat.completions.create(
            response_model=PolishedContent,
            messages=[
                {"role": "system", "content": self.prompts['polish_content']},
                {"role": "user", "content": f"请按照以下风格指导润色这段文本:\n风格指导: {style_prompt}\n\n原文:\n{content}"}
            ]
        )
        
        # 完成进度条
        progress_bar.update(80)
        progress_bar.close()
        
        return polished.content
    
    def save_novel(self, novel: Novel, filename: str = None) -> str:
        """保存小说到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = re.sub(r'[^\w\s]', '', novel.title).replace(' ', '_')
            filename = f"{safe_title}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(novel.model_dump(), ensure_ascii=False, indent=2))
        
        print(f"小说已保存至: {filename}")
        return filename
    
    def load_novel(self, filename: str) -> Novel:
        """从文件加载小说"""
        with open(filename, 'r', encoding='utf-8') as f:
            novel_data = json.load(f)
        
        return Novel(**novel_data)
    
    def reorganize_chapters(self, novel: Novel, chapter_order: List[int]) -> Novel:
        """重新组织章节顺序"""
        if max(chapter_order) >= len(novel.chapters) or min(chapter_order) < 0:
            raise ValueError("章节索引超出范围")
        
        reordered_novel = novel.model_copy()
        reordered_novel.chapters = [novel.chapters[i] for i in chapter_order]
        return reordered_novel
    
    def export_to_mdx(self, novel: Novel, filename: str = None) -> str:
        """将小说导出为MDX格式，非正文内容作为元数据"""
        if filename is None:
            safe_title = re.sub(r'[^\w\s]', '', novel.title).replace(' ', '_')
            filename = f"{safe_title}.mdx"
        
        # 准备元数据
        metadata = {
            "title": novel.title,
            "author": novel.author,
            "genre": novel.genre,
            "tags": novel.tags,
            "summary": novel.summary,
            "world_setting": {
                "name": novel.world_setting.name,
                "description": novel.world_setting.description,
                "rules": novel.world_setting.rules,
                "history": novel.world_setting.history
            },
            "characters": [character.model_dump() for character in novel.characters]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            # 写入YAML格式的元数据
            f.write("---\n")
            yaml_str = yaml.dump(metadata, allow_unicode=True, sort_keys=False)
            f.write(yaml_str)
            f.write("---\n\n")
            
            # 写入正文内容
            for i, chapter in enumerate(novel.chapters):
                f.write(f"# 第{i+1}章: {chapter.title}\n\n")
                f.write(f"{chapter.content}\n\n")
        
        print(f"小说已导出至: {filename}")
        return filename
    
    def extract_tags(self, novel: Novel) -> List[str]:
        """从小说中提取潜在标签"""
        print("正在提取小说标签...")
        
        # 创建进度条
        progress_bar = tqdm.tqdm(total=100, desc="正在提取标签", unit="%")
        progress_bar.update(10)
        
        prompt = f"""
        请从以下小说内容中提取关键标签，包括但不限于:
        - 主题标签
        - 情感标签
        - 场景标签
        - 写作风格标签
        
        小说标题: {novel.title}
        小说摘要: {novel.summary}
        世界观: {novel.world_setting.description}
        """
        
        # 使用instructor获取结构化输出
        extracted = self.client.chat.completions.create(
            response_model=ExtractedTags,
            messages=[
                {"role": "system", "content": self.prompts['extract_tags']},
                {"role": "user", "content": prompt}
            ]
        )
        
        # 更新进度条
        progress_bar.update(90)
        progress_bar.close()
        
        return extracted.tags
    
    def polish_chapter_interactive(self, novel: Novel, chapter_index: int) -> Novel:
        """交互式润色特定章节"""
        if chapter_index < 0 or chapter_index >= len(novel.chapters):
            print(f"错误: 章节索引 {chapter_index} 超出范围 (0-{len(novel.chapters)-1})")
            return novel
        
        chapter = novel.chapters[chapter_index]
        print(f"\n当前章节: 第{chapter_index+1}章 - {chapter.title}")
        print("\n前300个字符预览:")
        print(f"{chapter.content[:300]}...\n")
        
        style_prompt = input("请输入润色风格指导 (如: '使用更生动的描述', '增加紧张感'等): ")
        if not style_prompt:
            print("未提供风格指导，跳过润色")
            return novel
        
        # 润色内容
        polished_content = self.polish_content(chapter.content, style_prompt)
        
        # 创建修改后的小说副本
        polished_novel = novel.model_copy()
        polished_novel.chapters[chapter_index].content = polished_content
        
        print(f"\n润色完成! 章节'{chapter.title}'已更新。")
        return polished_novel


if __name__ == "__main__":
    # 定义提示词
    prompts = {
        "generate_novel": """
        你是一位专业的小说创作助手。根据用户的提示，创作一篇完整的中短篇小说。
        小说应该包含引人入胜的情节、鲜明的角色和丰富的世界观设定。
        
        生成的内容必须符合Pydantic模型的结构要求。
        
        特别注意:
        1. 每个章节的内容应当具有足够的细节和描写
        2. 每个章节应控制在2000-3000字左右
        3. 整个小说应该有完整的故事弧线，包括起承转合
        4. 角色应该有独特的性格特点和发展路径
        5. 世界设定应该具有一致性和合理性
        """,
        
        "polish_content": """
        你是一位专业的文学编辑。请根据用户提供的样式指导，对文本进行润色。
        保持原文的核心内容和意义，同时提升其文学质量、流畅度和吸引力。
        
        重要规则:
        1. 不要输出任何非正文内容，如"好的，请看我润色后的版本："等导语
        2. 只返回润色后的正文内容，不要添加任何解释或说明
        3. 保持与原文相似的篇幅，不要显著增加或减少内容长度
        4. 确保润色后的内容与前后章节的风格、情节保持一致性
        """,
        
        "extract_tags": """
        你是一位专业的文学分析师，善于从文本中提取关键标签。
        你需要从小说内容中提取最具代表性的标签，这些标签应该准确反映小说的主题、风格、情感和场景特点。
        
        注意事项:
        1. 标签应简洁明了，通常为1-3个词
        2. 每个标签应有明确的分类意义
        3. 避免过于宽泛的标签，如"小说"、"故事"等
        4. 提取10-15个最具代表性的标签
        5. 标签必须直接从小说内容中提取，不要创造未体现在内容中的标签
        """
    }
    
    # 初始化生成器
    generator = NovelGenerator(prompts)
    
    # 小说示例
    sample_prompt = "一个发生在2150年的科幻故事，描述一位年轻工程师发现了穿越时空的方法"
    
    # 交互式命令行
    while True:
        print("\n===== 中短篇小说生成器 =====")
        print("1. 创建新小说")
        print("2. 加载已有小说")
        print("3. 润色章节")
        print("4. 导出为MDX格式")
        print("5. 退出")
        
        choice = input("\n请选择操作 (1-5): ")
        
        if choice == "1":
            # 创建新小说
            print("\n==== 创建新小说 ====")
            prompt = input("请输入小说创作提示 (直接回车使用示例提示): ")
            if not prompt:
                prompt = sample_prompt
                print(f"使用示例提示: '{prompt}'")
            
            # 生成小说
            novel = generator.generate_novel(prompt)
            
            # 提取标签
            print("\n正在提取小说标签...")
            tags = generator.extract_tags(novel)
            novel.tags = tags
            print(f"提取到的标签: {', '.join(tags)}")
            
            # 保存小说
            filename = input("\n请输入保存文件名 (直接回车自动生成): ")
            if filename:
                json_file = generator.save_novel(novel, filename)
            else:
                json_file = generator.save_novel(novel)
            
            # 询问是否导出MDX
            if input("\n是否导出为MDX格式? (y/n): ").lower() == 'y':
                mdx_file = generator.export_to_mdx(novel, json_file.replace('.json', '.mdx'))
            
        elif choice == "2":
            # 加载已有小说
            print("\n==== 加载已有小说 ====")
            filename = input("请输入小说JSON文件路径: ")
            
            try:
                novel = generator.load_novel(filename)
                print(f"已加载小说《{novel.title}》")
                
                # 显示小说基本信息
                print(f"\n标题: {novel.title}")
                print(f"作者: {novel.author}")
                print(f"类型: {', '.join(novel.genre)}")
                print(f"标签: {', '.join(novel.tags)}")
                print(f"章节数: {len(novel.chapters)}")
                
                # 显示章节列表
                print("\n章节列表:")
                for i, chapter in enumerate(novel.chapters):
                    print(f"  {i}: {chapter.title}")
                
            except Exception as e:
                print(f"加载小说失败: {str(e)}")
                novel = None
                
        elif choice == "3":
            # 润色章节
            print("\n==== 润色章节 ====")
            
            if 'novel' not in locals() or novel is None:
                filename = input("请输入小说JSON文件路径: ")
                try:
                    novel = generator.load_novel(filename)
                    print(f"已加载小说《{novel.title}》")
                except Exception as e:
                    print(f"加载小说失败: {str(e)}")
                    continue
            
            # 显示章节列表
            print("\n章节列表:")
            for i, chapter in enumerate(novel.chapters):
                print(f"  {i}: {chapter.title}")
            
            while True:
                try:
                    chapter_index = int(input("\n请选择要润色的章节索引 (-1退出): "))
                    if chapter_index == -1:
                        break
                    
                    # 润色章节
                    novel = generator.polish_chapter_interactive(novel, chapter_index)
                    
                    # 询问是否保存
                    if input("\n是否保存修改? (y/n): ").lower() == 'y':
                        save_filename = input("请输入保存文件名 (直接回车覆盖原文件): ")
                        if not save_filename and 'filename' in locals():
                            save_filename = filename
                        
                        if save_filename:
                            generator.save_novel(novel, save_filename)
                    
                    # 询问是否继续
                    if input("\n是否继续润色其他章节? (y/n): ").lower() != 'y':
                        break
                        
                except ValueError:
                    print("无效输入，请输入数字索引")
                    
        elif choice == "4":
            # 导出为MDX
            print("\n==== 导出为MDX格式 ====")
            
            if 'novel' not in locals() or novel is None:
                filename = input("请输入小说JSON文件路径: ")
                try:
                    novel = generator.load_novel(filename)
                    print(f"已加载小说《{novel.title}》")
                except Exception as e:
                    print(f"加载小说失败: {str(e)}")
                    continue
            
            mdx_filename = input("请输入MDX文件名 (直接回车自动生成): ")
            if not mdx_filename:
                mdx_file = generator.export_to_mdx(novel)
            else:
                mdx_file = generator.export_to_mdx(novel, mdx_filename)
                
        elif choice == "5":
            # 退出
            print("\n感谢使用中短篇小说生成器！再见！")
            break
            
        else:
            print("\n无效选择，请输入1-5之间的数字。")