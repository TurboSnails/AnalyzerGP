"""
AI Agent 基类 - 统一的LLM调用接口
使用 litellm 支持: DeepSeek / Claude / GPT / 千问
"""
from typing import Dict, Any, Optional
from config import LLM_CONFIG


class BaseAgent:

    def __init__(self, role: str = "", system_prompt: str = ""):
        self.role          = role
        self.system_prompt = system_prompt
        self.model         = LLM_CONFIG["model"]
        self.temperature   = LLM_CONFIG["temperature"]
        self.max_tokens    = LLM_CONFIG["max_tokens"]

    def call_llm(self, prompt: str, context: str = "") -> str:
        try:
            import litellm

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            user_content = prompt
            if context:
                user_content = f"{prompt}\n\n---\n以下是相关数据:\n{context}"

            messages.append({"role": "user", "content": user_content})

            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=LLM_CONFIG.get("api_key"),
            )
            return response.choices[0].message.content

        except ImportError:
            return "[错误] 请安装 litellm: pip install litellm"
        except Exception as e:
            return f"[LLM调用失败] {str(e)}"


class BullAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            role="多头分析师",
            system_prompt=(
                "你是一位专业的多头分析师。你的任务是基于数据找出公司的核心投资价值，"
                "包括被低估的优势、潜在的催化剂、以及市场可能忽视的正面因素。"
                "你的分析必须基于事实和数据，不能凭空乐观。使用中文回答。"
            ),
        )

    def analyze(self, company: str, data_context: str) -> str:
        prompt = (
            f"请对 {company} 进行多头分析，重点回答:\n"
            f"1. 当前估值是否被低估？为什么？\n"
            f"2. 未来12-18个月最可能的正面催化剂是什么？\n"
            f"3. 市场可能忽视了哪些正面因素？\n"
            f"4. 如果要买入，最有力的三个理由是什么？\n"
            f"请基于提供的数据分析，结论务必有数据支撑。"
        )
        return self.call_llm(prompt, data_context)


class BearAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            role="空头分析师",
            system_prompt=(
                "你是一位专业的空头分析师，你的职责是找出投资论点中的漏洞和风险。"
                "你不需要客气，必须给出最犀利的质疑。你的目标是帮助投资者避免亏损。"
                "你的每个反对理由都必须基于真实的商业逻辑，不能是泛泛而谈的风险提示。"
                "使用中文回答。"
            ),
        )

    def analyze(self, company: str, bull_case: str, data_context: str) -> str:
        prompt = (
            f"多头分析师认为 {company} 值得投资，理由如下:\n"
            f"{bull_case}\n\n"
            f"请作为空头分析师，给出最有力的5个反对理由:\n"
            f"1. 基于真实商业逻辑，不能重复已提到的风险\n"
            f"2. 如果你要做空这只股票，最核心的逻辑是什么？\n"
            f"3. 18个月后股价没有反弹，最可能的原因是什么？\n"
            f"4. 最坏情况下会发生什么？\n"
            f"5. 市场上更聪明的机构为什么没有大举买入？"
        )
        return self.call_llm(prompt, data_context)


class FirstPrinciplesAgent(BaseAgent):
    """Phase 3: 第一性原理分析 Agent
    穿透表象，分析商业模式本质、需求持久性、核心壁垒
    """

    def __init__(self):
        super().__init__(
            role="第一性原理分析师",
            system_prompt=(
                "你是一位用第一性原理思考的投资分析师。你的任务是穿透财务数字和行业术语，"
                "回到最本质的问题：这家公司解决了什么真实需求？这个需求是否持久？"
                "公司凭什么能长期赚钱而不被竞争者消灭？\n\n"
                "分析原则:\n"
                "1. 从需求端出发，不从供给端（公司）出发\n"
                "2. 问'为什么'至少三层，找到真正的因果链\n"
                "3. 明确区分：什么是事实，什么是假设，什么是风险\n"
                "4. 结论必须可证伪——说明什么情况下你的分析会被证伪\n"
                "使用中文回答。"
            ),
        )

    def analyze(self, company: str, data_context: str) -> str:
        prompt = (
            f"请用第一性原理分析 {company}，回答以下问题:\n\n"
            f"**1. 本质需求**\n"
            f"这家公司满足了用户/客户什么本质需求？这个需求是刚需还是可选？"
            f"10年后这个需求还存在吗？\n\n"
            f"**2. 盈利逻辑**\n"
            f"公司为什么能赚钱？是因为垄断、效率、品牌还是其他？"
            f"这个盈利来源是否可持续？竞争者为什么没有复制？\n\n"
            f"**3. 护城河本质**\n"
            f"用一句话描述这家公司最核心的竞争壁垒。"
            f"这个壁垒是在加深还是在侵蚀？\n\n"
            f"**4. 最大的假设**\n"
            f"你的投资逻辑中最重要的三个假设是什么？"
            f"如果哪个假设被证伪，整个逻辑就会崩塌？\n\n"
            f"**5. 可证伪条件**\n"
            f"列出2-3个具体指标，如果这些指标出现，说明投资逻辑失效，应该立即重新评估。"
        )
        return self.call_llm(prompt, data_context)


class JudgeAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            role="投资决策裁判",
            system_prompt=(
                "你是一位经验丰富的投资组合经理，遵循以下逻辑驱动框架做决策:\n\n"
                "【仓位原则】\n"
                "- 单标的硬性上限: 总资产10%\n"
                "- Alpha层合计上限: 25%（熊市压缩至12%）\n"
                "- 不使用凯利公式精确计算（输入参数无法客观量化）\n"
                "- 高确信度→靠近10%上限；低确信度→靠近5%下限\n\n"
                "【退出原则】\n"
                "- 逻辑失效（营收减速、毛利下滑、核心指标恶化）→立即清仓\n"
                "- 价格大涨但逻辑已失效 → 卖出，这是风险到来不是奖励\n"
                "- 18个月逻辑成立但市场无反应 → 重新评估\n"
                "- 24个月无效率 → 减仓至1/2，释放资金\n\n"
                "【框架优先级】\n"
                "- 宏观状态(Layer 0)优先于个股仓位决策\n"
                "- 熊市中段暂停新开Alpha仓位\n"
                "- 逻辑是否成立比价格涨跌更重要\n\n"
                "你必须给出明确的行动建议，不能模棱两可。使用中文回答。"
            ),
        )

    def decide(
        self,
        company: str,
        bull_case: str,
        bear_case: str,
        quant_data: str,
        macro_context: str,
        position_context: str = "",
    ) -> str:
        prompt = (
            f"请对 {company} 做出最终投资决策。\n\n"
            f"## 宏观环境 (Layer 0)\n{macro_context}\n\n"
            f"## 量化数据\n{quant_data}\n\n"
            f"## 系统仓位建议\n{position_context}\n\n"
            f"## 多头观点\n{bull_case}\n\n"
            f"## 空头观点\n{bear_case}\n\n"
            f"请给出:\n"
            f"1. **综合评分**: 满分10分，说明评分依据\n"
            f"2. **投资结论**: 买入/观望/回避（一句话核心理由）\n"
            f"3. **建议首批仓位**: 占总资产百分比，说明为何选这个档位\n"
            f"4. **逻辑退出条件**: 什么指标恶化时必须离场（非价格止损）\n"
            f"5. **时间窗口**: 逻辑验证的关键时间节点\n"
            f"6. **最需跟踪的3个指标**: 每季度必看的逻辑验证点\n"
        )
        return self.call_llm(prompt)
