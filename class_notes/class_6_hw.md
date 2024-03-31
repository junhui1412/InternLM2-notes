# 基础作业

## Lagent Web Demo

### 使用 LMDeploy 部署

由于 Lagent 的 Web Demo 需要用到 LMDeploy 所启动的 api_server，因此我们首先按照下图指示在 vscode terminal 中执行如下代码使用 LMDeploy 启动一个 api_server。

```sh
lmdeploy serve api_server \
    /root/models/internlm2-chat-7b \
    --server-name 127.0.0.1 \
    --model-name internlm2-chat-7b \
    --cache-max-entry-count 0.1
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo1.png)

### 启动并使用 Lagent Web Demo

新建一个 terminal 以启动 Lagent Web Demo。在新建的 terminal 中执行如下指令：

```sh
cd /root/agent/lagent/examples
streamlit run internlm2_agent_web_demo.py --server.address 127.0.0.1 --server.port 7860
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo2.png)

### 映射本地端口

在**本地**进行端口映射，将 LMDeploy api_server 的23333端口以及 Lagent Web Demo 的7860端口映射到本地。

```sh
ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 40165
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo3.png)

### [lagent-web](http://localhost:7860/)

![LagentWebDemo4](../attachment/InternLM2_homework6.assets/LagentWebDemo4.jpeg)

### 搜索论文

搜索 InternLM2 Technical Report

![LagentWebDemo5](../attachment/InternLM2_homework6.assets/LagentWebDemo5.jpeg)

搜索 LlaMa2

![LagentWebDemo6](../attachment/InternLM2_homework6.assets/LagentWebDemo6.jpeg)

搜索 Imagen

![LagentWebDemo7](../attachment/InternLM2_homework6.assets/LagentWebDemo7.jpeg)

## AgentLego

### 下载 demo 文件

```sh
cd /root/agent
wget http://download.openmmlab.com/agentlego/road.jpg
```

![](../attachment/InternLM2_homework6.assets/AgentLego1.png)

### 安装依赖

由于 AgentLego 在安装时并不会安装某个特定工具的依赖，因此我们接下来准备安装目标检测工具运行时所需依赖。

AgentLego 所实现的目标检测工具是基于 mmdet (MMDetection) 算法库中的 RTMDet-Large 模型，因此我们首先安装 mim，然后通过 mim 工具来安装 mmdet。

```sh
pip install openmim==0.3.9
mim install mmdet==3.3.0
```

![](../attachment/InternLM2_homework6.assets/AgentLego2.png)

![](../attachment/InternLM2_homework6.assets/AgentLego3.png)

### 创建指定代码

然后通过 `touch /root/agent/direct_use.py`（大小写敏感）的方式在 /root/agent 目录下新建 direct_use.py 以直接使用目标检测工具，direct_use.py 的代码如下：

```python
import re

import cv2
from agentlego.apis import load_tool

# load tool
tool = load_tool('ObjectDetection', device='cuda')

# apply tool
visualization = tool('/root/agent/road.jpg')
print(visualization)

# visualize
image = cv2.imread('/root/agent/road.jpg')

preds = visualization.split('\n')
pattern = r'(\w+) \((\d+), (\d+), (\d+), (\d+)\), score (\d+)'

for pred in preds:
    name, x1, y1, x2, y2, score = re.match(pattern, pred).groups()
    x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), int(score)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, f'{name} {score}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

cv2.imwrite('/root/agent/road_detection_direct.jpg', image)
```

### 运行代码

```sh
cd /root/agent
python direct_use.py
```

![](../attachment/InternLM2_homework6.assets/AgentLego4.png)

效果如图所示，左侧为原图，右侧为绘制的框

![](../attachment/InternLM2_homework6.assets/AgentLego5.png)

# 进阶作业

## AgentLego WebUI 使用

### 修改相关文件

由于 AgentLego 算法库默认使用 InternLM2-Chat-20B 模型，因此我们首先需要修改 /root/agent/agentlego/webui/modules/agents/lagent_agent.py 文件的第 105行位置，将 internlm2-chat-20b 修改为 internlm2-chat-7b，即

![](../attachment/InternLM2_homework6.assets/AgentLegoWebUI1.png)

### 使用 LMDeploy 部署

由于 Lagent 的 Web Demo 需要用到 LMDeploy 所启动的 api_server，因此我们首先按照下图指示在 vscode terminal 中执行如下代码使用 LMDeploy 启动一个 api_server。

```sh
lmdeploy serve api_server \
    /root/models/internlm2-chat-7b \
    --server-name 127.0.0.1 \
    --model-name internlm2-chat-7b \
    --cache-max-entry-count 0.1
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo1.png)

### 启动 AgentLego WebUI

新建一个 terminal 以启动 AgentLego WebUI。

```sh
cd /root/agent/agentlego/webui
python one_click.py
```

![](../attachment/InternLM2_homework6.assets/AgentLegoWebUI2.png)

### 映射本地端口

在**本地**进行端口映射，将 LMDeploy api_server 的23333端口以及 Lagent Web Demo 的7860端口映射到本地。

```sh
ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 40165
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo3.png)

### [agentlego-web](http://localhost:7860/)

配置agent

![AgentLegoWebUI3](../attachment/InternLM2_homework6.assets/AgentLegoWebUI3.jpeg)

配置tool

![AgentLegoWebUI4](../attachment/InternLM2_homework6.assets/AgentLegoWebUI4.jpeg)

只选择 ObjectDetection

![](../attachment/InternLM2_homework6.assets/AgentLegoWebUI5.png)

检测

![AgentLegoWebUI6](../attachment/InternLM2_homework6.assets/AgentLegoWebUI6.jpeg)

![AgentLegoWebUI7](../attachment/InternLM2_homework6.assets/AgentLegoWebUI7.jpeg)

## [用 Lagent 自定义工具](https://github.com/InternLM/Tutorial/blob/camp2/agent/lagent.md#2-用-lagent-自定义工具)

我们将基于 Lagent 自定义一个工具。Lagent 中关于工具部分的介绍文档位于 https://lagent.readthedocs.io/zh-cn/latest/tutorials/action.html 。使用 Lagent 自定义工具主要分为以下几步：

1. 继承 BaseAction 类
2. 实现简单工具的 run 方法；或者实现工具包内每个子工具的功能
3. 简单工具的 run 方法可选被 tool_api 装饰；工具包内每个子工具的功能都需要被 tool_api 装饰

下面我们将实现一个调用和风天气 API 的工具以完成实时天气查询的功能。

### 创建工具文件

首先通过 `touch /root/agent/lagent/lagent/actions/weather.py`（大小写敏感）新建工具文件，该文件内容如下：

```python
import json
import os
import requests
from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

class WeatherQuery(BaseAction):
    """Weather plugin for querying weather information."""
    
    def __init__(self,
                 key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        key = os.environ.get('WEATHER_API_KEY', key)
        if key is None:
            raise ValueError(
                'Please set Weather API key either in the environment '
                'as WEATHER_API_KEY or pass it as `key`')
        self.key = key
        self.location_query_url = 'https://geoapi.qweather.com/v2/city/lookup'
        self.weather_query_url = 'https://devapi.qweather.com/v7/weather/now'

    @tool_api
    def run(self, query: str) -> ActionReturn:
        """一个天气查询API。可以根据城市名查询天气信息。
        
        Args:
            query (:class:`str`): The city name to query.
        """
        tool_return = ActionReturn(type=self.name)
        status_code, response = self._search(query)
        if status_code == -1:
            tool_return.errmsg = response
            tool_return.state = ActionStatusCode.HTTP_ERROR
        elif status_code == 200:
            parsed_res = self._parse_results(response)
            tool_return.result = [dict(type='text', content=str(parsed_res))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = str(status_code)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
    
    def _parse_results(self, results: dict) -> str:
        """Parse the weather results from QWeather API.
        
        Args:
            results (dict): The weather content from QWeather API
                in json format.
        
        Returns:
            str: The parsed weather results.
        """
        now = results['now']
        data = [
            f'数据观测时间: {now["obsTime"]}',
            f'温度: {now["temp"]}°C',
            f'体感温度: {now["feelsLike"]}°C',
            f'天气: {now["text"]}',
            f'风向: {now["windDir"]}，角度为 {now["wind360"]}°',
            f'风力等级: {now["windScale"]}，风速为 {now["windSpeed"]} km/h',
            f'相对湿度: {now["humidity"]}',
            f'当前小时累计降水量: {now["precip"]} mm',
            f'大气压强: {now["pressure"]} 百帕',
            f'能见度: {now["vis"]} km',
        ]
        return '\n'.join(data)

    def _search(self, query: str):
        # get city_code
        try:
            city_code_response = requests.get(
                self.location_query_url,
                params={'key': self.key, 'location': query}
            )
        except Exception as e:
            return -1, str(e)
        if city_code_response.status_code != 200:
            return city_code_response.status_code, city_code_response.json()
        city_code_response = city_code_response.json()
        if len(city_code_response['location']) == 0:
            return -1, '未查询到城市'
        city_code = city_code_response['location'][0]['id']
        # get weather
        try:
            weather_response = requests.get(
                self.weather_query_url,
                params={'key': self.key, 'location': city_code}
            )
        except Exception as e:
            return -1, str(e)
        return weather_response.status_code, weather_response.json()
```

![](../attachment/InternLM2_homework6.assets/weather1.png)

### 获取api key

![weather2](../attachment/InternLM2_homework6.assets/weather2.jpeg)

![weather3](../attachment/InternLM2_homework6.assets/weather3.jpeg)

### 使用 LMDeploy 部署

使用 LMDeploy 启动一个 api_server。

```sh
lmdeploy serve api_server \
    /root/models/internlm2-chat-7b \
    --server-name 127.0.0.1 \
    --model-name internlm2-chat-7b \
    --cache-max-entry-count 0.1
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo1.png)

### 启动天气服务

```sh
export WEATHER_API_KEY=API KEY 

cd /root/Tutorial/agent
streamlit run internlm2_weather_web_demo.py --server.address 127.0.0.1 --server.port 7860
```

![](../attachment/InternLM2_homework6.assets/weather4.png)

### 映射本地端口

在**本地**进行端口映射，将 LMDeploy api_server 的23333端口以及 Lagent Web Demo 的7860端口映射到本地。

```sh
ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 40165
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo3.png)

### [lagent-web](http://localhost:7860/)

![weather5](../attachment/InternLM2_homework6.assets/weather5.jpeg)

### 查询天气

查询杭州天气

![weather6](../attachment/InternLM2_homework6.assets/weather6.jpeg)

查询北京天气

![weather7](../attachment/InternLM2_homework6.assets/weather7.jpeg)

查询上海天气

![weather8](../attachment/InternLM2_homework6.assets/weather8.jpeg)

## [用 AgentLego 自定义工具](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md#3-用-agentlego-自定义工具)

在本节中，我们将基于 AgentLego 构建自己的自定义工具。AgentLego 在这方面提供了较为详尽的文档，文档地址为 https://agentlego.readthedocs.io/zh-cn/latest/modules/tool.html 。自定义工具主要分为以下几步：

1. 继承 BaseTool 类
2. 修改 default_desc 属性（工具功能描述）
3. 如有需要，重载 setup 方法（重型模块延迟加载）
4. 重载 apply 方法（工具功能实现）

其中第一二四步是必须的步骤。下面我们将实现一个调用 MagicMaker 的 API 以实现图像生成的工具。

MagicMaker 是汇聚了优秀 AI 算法成果的免费 AI 视觉素材生成与创作平台。主要提供图像生成、图像编辑和视频生成三大核心功能，全面满足用户在各种应用场景下的视觉素材创作需求。体验更多功能可以访问 https://magicmaker.openxlab.org.cn/home 。

### 创建工具文件

首先通过 `touch /root/agent/agentlego/agentlego/tools/magicmaker_image_generation.py`（大小写敏感）的方法新建工具文件。该文件的内容如下：

```python
import json
import requests

import numpy as np

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from .base import BaseTool


class MagicMakerImageGeneration(BaseTool):

    default_desc = ('This tool can call the api of magicmaker to '
                    'generate an image according to the given keywords.')

    styles_option = [
        'dongman',  # 动漫
        'guofeng',  # 国风
        'xieshi',   # 写实
        'youhua',   # 油画
        'manghe',   # 盲盒
    ]
    aspect_ratio_options = [
        '16:9', '4:3', '3:2', '1:1',
        '2:3', '3:4', '9:16'
    ]

    @require('opencv-python')
    def __init__(self,
                 style='guofeng',
                 aspect_ratio='4:3'):
        super().__init__()
        if style in self.styles_option:
            self.style = style
        else:
            raise ValueError(f'The style must be one of {self.styles_option}')
        
        if aspect_ratio in self.aspect_ratio_options:
            self.aspect_ratio = aspect_ratio
        else:
            raise ValueError(f'The aspect ratio must be one of {aspect_ratio}')

    def apply(self,
              keywords: Annotated[str,
                                  Info('A series of Chinese keywords separated by comma.')]
        ) -> ImageIO:
        import cv2
        response = requests.post(
            url='https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate',
            data=json.dumps({
                "official": True,
                "prompt": keywords,
                "style": self.style,
                "poseT": False,
                "aspectRatio": self.aspect_ratio
            }),
            headers={'content-type': 'application/json'}
        )
        image_url = response.json()['data']['imgUrl']
        image_response = requests.get(image_url)
        image = cv2.imdecode(np.frombuffer(image_response.content, np.uint8), cv2.IMREAD_COLOR)
        return ImageIO(image)
```

![ImageGeneration1](../attachment/InternLM2_homework6.assets/ImageGeneration1.png)

### 注册新工具

修改 /root/AgentLego/agentlego/agentlego/tools/__init__.py 文件，将我们的工具注册在工具列表中。

如下所示，我们将 MagicMakerImageGeneration 通过 from .magicmaker_image_generation import MagicMakerImageGeneration 导入到了文件中，并且将其加入了 __all__ 列表中

```python
from .base import BaseTool
from .calculator import Calculator
from .func import make_tool
from .image_canny import CannyTextToImage, ImageToCanny
from .image_depth import DepthTextToImage, ImageToDepth
from .image_editing import ImageExpansion, ImageStylization, ObjectRemove, ObjectReplace
from .image_pose import HumanBodyPose, HumanFaceLandmark, PoseToImage
from .image_scribble import ImageToScribble, ScribbleTextToImage
from .image_text import ImageDescription, TextToImage
from .imagebind import AudioImageToImage, AudioTextToImage, AudioToImage, ThermalToImage
from .object_detection import ObjectDetection, TextToBbox
from .ocr import OCR
from .scholar import *  # noqa: F401, F403
from .search import BingSearch, GoogleSearch
from .segmentation import SegmentAnything, SegmentObject, SemanticSegmentation
from .speech_text import SpeechToText, TextToSpeech
from .translation import Translation
from .vqa import VQA
from .magicmaker_image_generation import MagicMakerImageGeneration

__all__ = [
    'CannyTextToImage', 'ImageToCanny', 'DepthTextToImage', 'ImageToDepth',
    'ImageExpansion', 'ObjectRemove', 'ObjectReplace', 'HumanFaceLandmark',
    'HumanBodyPose', 'PoseToImage', 'ImageToScribble', 'ScribbleTextToImage',
    'ImageDescription', 'TextToImage', 'VQA', 'ObjectDetection', 'TextToBbox', 'OCR',
    'SegmentObject', 'SegmentAnything', 'SemanticSegmentation', 'ImageStylization',
    'AudioToImage', 'ThermalToImage', 'AudioImageToImage', 'AudioTextToImage',
    'SpeechToText', 'TextToSpeech', 'Translation', 'GoogleSearch', 'Calculator',
    # 'BaseTool', 'make_tool', 'BingSearch',
    'BaseTool', 'make_tool', 'BingSearch',, 'MagicMakerImageGeneration'
]
```

![](../attachment/InternLM2_homework6.assets/ImageGeneration2.png)

### 使用 LMDeploy 部署

使用 LMDeploy 启动一个 api_server。

```sh
lmdeploy serve api_server \
    /root/models/internlm2-chat-7b \
    --server-name 127.0.0.1 \
    --model-name internlm2-chat-7b \
    --cache-max-entry-count 0.1
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo1.png)

### 启动 AgentLego WebUI

新建一个 terminal 以启动 AgentLego WebUI。

```sh
cd /root/agent/agentlego/webui
python one_click.py
```

![](../attachment/InternLM2_homework6.assets/ImageGeneration3.png)

### 映射本地端口

在**本地**进行端口映射，将 LMDeploy api_server 的23333端口以及 Lagent Web Demo 的7860端口映射到本地。

```sh
ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 40165
```

![](../attachment/InternLM2_homework6.assets/LagentWebDemo3.png)

### [agentlego-web](http://localhost:7860/)

配置agent

![](../attachment/InternLM2_homework6.assets/ImageGeneration4.png)

添加 MagicMakerImageGeneration

![](../attachment/InternLM2_homework6.assets/ImageGeneration5.png)

只选择 MagicMakerImageGeneration

![](../attachment/InternLM2_homework6.assets/ImageGeneration6.png)

生成动漫图片

![ImageGeneration7](../attachment/InternLM2_homework6.assets/ImageGeneration7.jpeg)

生成油画

![ImageGeneration8](../attachment/InternLM2_homework6.assets/ImageGeneration8.jpeg)

生成写实图片

![ImageGeneration9](../attachment/InternLM2_homework6.assets/ImageGeneration9.jpeg)