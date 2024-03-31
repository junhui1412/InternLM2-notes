# Class 1
## 总结  
本次视频介绍了书生·浦语大模型的全链路开源体系，重点在于通用人工智能的发展趋势，特别是从特定任务模型向通用大模型的转变，以及书生模型在7月、9月和1月的升级，包括支持多模态、8K语境和不同尺寸的模型，以及在语言建模能力、对话交互和智能体框架方面的提升。  
  
## 要点  
- 通用人工智能的发展方向：从单一任务模型转向通用大模型，解决多种任务和模态。  
- 书生·浦语模型升级：7月升级支持8K语境和工具体系，8月发布对话模型和智能体框架，9月发布中等尺寸模型与优化工具链。  
- 英特尔M2开源：提升模型性能，支持复杂场景，提供不同尺寸的模型以适应不同需求。  
- 模型能力亮点：长上下文理解、对话与创作、数学能力等，例如通过模型进行行程规划和情感对话。  
- 英特尔M2的优化：数据清洗、高质量语料和新数据补全提升模型性能。  
- 下游任务性能提升：模型使用更少数据也能达到上一代效果，整体性能增强。  
- 开源工具体系：覆盖数据到预训练、微调、部署和评测等全流程，如INTETRAIN、XTA、i M Deploy等。  
- 数据集：提供丰富多样的数据，支持数据清洗、安全处理和公开使用。  
- 性能评测与差距：大模型整体能力仍有提升空间，尤其在理科能力上，中文场景下国内模型表现出色。  
- 部署解决方案：i M Deploy支持模型轻量化、量化和推理服务，与评测工具无缝对接。

# InternLM2

**数据准备：**

- **文本数据：** 广泛的过滤和重复数据消除过程确保了用于预培训的高质量、安全的文本数据集，主要来源于网页、论文、专利和书籍。
- **代码数据：** 将基于规则和基于模型的评分相结合的混合方法用于代码数据的质量过滤，重点是保持上下文完整性并将代码与自然语言交织。
- **长上下文数据：** 使用统计和困惑过滤器来去除低质量的长文本数据，确保扩展上下文窗口内的一致性和相关性。

**训练技巧：**

- **长上下文训练：** InternetLM2最初在4k tokens 上下文上进行训练，然后过渡到高质量的32k文本，从而实现高效的长期依赖性捕获。这种方法与组查询注意力（GQA）和位置编码外推相结合，使InternLM2能够在需要长时间理解上下文的任务中表现出色。
- **COOL RLHF:** 这种新颖的方法利用条件奖励模型来解决相互冲突的人类偏好，并减少奖励黑客攻击。通过根据特定条件动态调整对不同偏好的关注，COOL RLHF显著提高了模型与人类期望的一致性。
- **工具增强培训：** InternetLM2集成了通用工具调用和代码解释器集成，增强了其解决复杂问题的能力，尤其是在数学和数据分析任务中。

**评估结果：**

- **综合考试：** InternetLM2在各种与考试相关的数据集中优于其他开源LLM，包括MMLU、CMMLU、C-Eval、AGIEval和GAOKAO Bench。
- **语言和知识：** 在涉及语言理解和知识应用的任务中表现领先，如TriviaQA、NaturalQuestions、C3、RACE High和FLORES。
- **推理和数学：** 在推理和数学基准测试方面取得了最先进的成绩，包括WinoGrande、HellaSwag、BBH、GSM8K Test、MATH、TheoremQA和MathBench。
- **编码：** 在代码生成任务方面表现出非凡的熟练度，超过了以前在HumanEval、MBPP和MBPP-CN基准测试上最先进的模型。
- **长上下文建模：** 在L-Eval和LongBench基准测试上表现出强大的性能，展示了其处理长而复杂文本的能力。
- **工具利用率：** 在利用外部工具和API时，特别是在解决数学问题和执行数据分析任务方面，显示出显著的改进。
- **比对：** 在AlpacaEval、MTBench、CompassArena、AlignBench和IFEval等主观比对数据集上实现SOTA或接近SOTA的结果，显示出与人类偏好的强烈一致性。