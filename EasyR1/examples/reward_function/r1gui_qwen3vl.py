import re
import json
from typing import Any
import json
import random


def safe_unicode_unescape(s: str) -> str:
    # 只处理看起来真的有转义的情况
    if r'\u' in s or r'\x' in s:
        try:
            return s.encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return s
    return s

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def extract_action(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r'"action_type":\s*"(.*?)"'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no action"

def extract_input_text(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r'"value":\s*"(.*?)"'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no input text"

def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'"point_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0], False
    except:
        return [0, 0], False
    


def r1gui_format_reward(predict_str: str) -> float:
    """
    检查 predict_str 是否符合 <thinking></thinking><answer></answer> 的格式，
    并验证 <answer> 中的内容是否符合
    {"action_type": "...", "action_description": "...", "value": "...", "point_2d": [x, y] 或 "none"} 的格式。
    """
    # 1. 外部结构
    # outer_pattern = re.compile(r"<thinking>.*?</thinking>\s*<answer>.*?</answer>", re.DOTALL)
    # if not outer_pattern.match(predict_str):
    #     return 0.0
    outer_pattern = re.compile(
        r"^(?:<thinking>.*?</thinking>\s*)?<answer>.*?</answer>$",
        re.DOTALL
    )
    if not outer_pattern.match(predict_str):
        return 0.0

    # 2. 抽取 <answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1).strip()

    # 3. 解析 JSON
    try:
        actions = json.loads(answer_content)
    except Exception as e:
        at_match = re.search(r'"action_type"\s*:\s*"([^"]*)"', answer_content)
        # 提取 action_description (字符串，允许中间有换行，使用 DOTALL 假如 value 跨行)
        desc_match = re.search(r'"action_description"\s*:\s*"([^"]*)"', answer_content)
        # # 提取 value (字符串)
        val_match = re.search(r'"value"\s*:\s*"([^"]*)"', answer_content)
        # 提取 point_2d (匹配方括号内的内容)
        point, pt_match = extract_coord(answer_content)
        
        if at_match and desc_match and val_match and pt_match:
            actions = {
                "action_type": at_match.group(1),
                "action_description": desc_match.group(1),
                "value": val_match.group(1),
                "point_2d": point
            }
        else:
            # 连正则都匹配不到关键字段，彻底放弃
            print("JSON parse error: ", answer_content)
            return 0.0

    # 4. 必须是 dict
    if not isinstance(actions, dict):
        return 0.0

    # 5. 必要字段
    required_keys = {"action_type", "action_description", "value", "point_2d"}
    if not required_keys.issubset(actions):
        return 0.0

    # 6. 类型检查
    if not isinstance(actions["action_type"], str):
        return 0.0
    if not isinstance(actions["action_description"], str):
        return 0.0
    if not isinstance(actions["value"], str):
        return 0.0

    # 7. point_2d：可以是 [x, y] 或 "none"/"None"
    point = actions["point_2d"]
    if isinstance(point, list):
        if len(point) != 2 or not all(isinstance(v, int) for v in point):
            return 0.0
    elif isinstance(point, str):
        if point.lower() != "none":
            return 0.0
    else:
        return 0.0

    # 全部通过
    return 1.0


def r1gui_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    比较 predict_str 和 ground_truth 中的动作和参数是否一致。
    """
    try:
        # 提取 ground_truth 的动作和参数
        gt_action=ground_truth['gt_action'].lower()
        
        if 'gt_bbox' in ground_truth:
            gt_point_2d=ground_truth['gt_bbox']
            gt_point_2d = [coord / 1000 for coord in gt_point_2d]
        elif 'gt_point_2d' in ground_truth:
            gt_point_2d=ground_truth['gt_point_2d']
        else:
            raise ValueError("No ground truth coordinates found.")
        gt_input_text=ground_truth['gt_input_text'].lower()
        gt_input_text = safe_unicode_unescape(gt_input_text)
        if 'image_width_new' not in ground_truth or 'image_height_new' not in ground_truth:
            width, height=ground_truth['image_width'], ground_truth['image_height']
        else:
            width, height=ground_truth['image_width_new'], ground_truth['image_height_new']
        pred_action=extract_action(predict_str).lower()
        pred_input_text=extract_input_text(predict_str).lower()
        pred_bbox,_=extract_coord(predict_str)
        pred_bbox = [coord / 1000 for i, coord in enumerate(pred_bbox)]

        # map actions
        action_map = {
            'long_press': 'longpress',
            'open_app': 'openapp',
            'type': 'write',
            'press_back': 'navigateback',
            'press_home': 'navigatehome',
            'pressback': 'navigateback',
            'presshome': 'navigatehome',
        }
        if gt_action in action_map:
            gt_action = action_map[gt_action]

        if pred_action!=gt_action:
            return 0.0
        
        if gt_action in ["click", "longpress"]:

            if len(gt_point_2d)==2:
                distance = (pred_bbox[0]-gt_point_2d[0])**2+(pred_bbox[1]-gt_point_2d[1])**2
                if distance < 0.05**2:
                    return 1.0
                else:
                    return 0.0
            elif len(gt_point_2d)==4:
                if (gt_point_2d[0]<pred_bbox[0]<gt_point_2d[2]) and (gt_point_2d[1]<pred_bbox[1]<gt_point_2d[3]):
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
        elif gt_action in ['type', 'write', 'select','scroll', 'openapp', 'swipe', 'terminate', 'answer']:
            if calculate_f1_score(pred_input_text,gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        else:
            return 1.0

    except Exception as e:
        return 0.0
    
def compute_score(reward_inputs: dict[str, Any], format_weight: float = 0.1) -> dict[str, float]:
    if not isinstance(reward_inputs, dict):
        raise ValueError("Please use `reward_type=sequential` for r1v reward function.")


    format_score = r1gui_format_reward(reward_inputs["response"])
    accuracy = r1gui_accuracy_reward(reward_inputs["response"], reward_inputs["ground_truth"])
    if random.random()<0.01:
        width, height=reward_inputs["ground_truth"]['image_width_new'], reward_inputs["ground_truth"]['image_height_new']
        gt_bbox = reward_inputs["ground_truth"]['gt_bbox']
        gt_bbox = [x / 1000 * width if i % 2 == 0 else x / 1000 * height for i, x in enumerate(gt_bbox)]
        print(reward_inputs["response"], reward_inputs["ground_truth"])
        # print(f"GT coord: ({ground_truth_x}, {ground_truth_y})")
        print(f"GT bbox: {gt_bbox}")
        print(f"Format score: {format_score}, Accuracy score: {accuracy}")

    if '<thinking>' in reward_inputs["response"] and '</thinking>' in reward_inputs["response"]:
        use_reasoning = 1.0
    else:
        use_reasoning = 0.0
    
    is_grounding = reward_inputs["ground_truth"].get('is_grounding', False)
    if not is_grounding and use_reasoning == 0.0:
        # if a reasoning task but no reasoning used, set format score to 0
        format_score = 0.0
    elif is_grounding and use_reasoning == 1.0:
        format_score = 0.0

    return {
        "overall": (1 - format_weight) * accuracy + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy,
        "use_reasoning": use_reasoning
    }

