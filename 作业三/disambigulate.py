import math
from collections import Counter
import os
import re
import time
def _parse_people_daily_line(line_text):
    """
    解析人民日报语料库格式的单行文本。
    例如: "本报/rz 北京/ns ... [24/m 个/q]/mq ... 。/w"
    返回一个词语列表。
    """
    words = []
    # 将多个空格替换为单个空格，然后按空格分割
    cleaned_line = re.sub(r'\s+', ' ', line_text.strip())
    tokens = cleaned_line.split(' ')

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # 检查形如 "[词1/词性1 词2/词性2]/短语词性" 的模式
        # 例如: "[24/m 个/q]/mq"
        match_bracketed_phrase = re.match(r'\[(.*?)\]/([a-zA-Z0-9_]+)$', token)
        if match_bracketed_phrase:
            phrase_content = match_bracketed_phrase.group(1)
            # phrase_content 类似于 "词1/词性1 词2/词性2"
            inner_tokens = phrase_content.split(' ')
            for inner_token in inner_tokens:
                inner_token = inner_token.strip()
                if inner_token:
                    # 从内部词块中移除词性标记
                    word_only = inner_token.rsplit('/', 1)[0] if '/' in inner_token else inner_token
                    words.append(word_only)
        else:
            # 普通的 "词/词性" 或可能只是一个单独的词（例如标点符号）
            # 移除词性标记
            word_only = token.rsplit('/', 1)[0] if '/' in token else token
            words.append(word_only)
    return words

def load_corpus_from_directory(base_dir_path):
    """
    从指定目录加载语料库。
    遍历目录及其子目录下的所有 .txt 文件，解析它们的内容。
    :param base_dir_path: 语料库的根目录路径 (str)
    :return: 一个句子列表，其中每个句子是词的列表 (list of list of str)
    """
    corpus_sentences = []
    
    # 将相对路径转换为绝对路径，以便os.path.isdir等函数能正确工作
    # __file__ 是当前脚本的路径
    # os.path.dirname(__file__) 是当前脚本所在的目录
    # 如果base_dir_path已经是绝对路径，os.path.abspath会返回它本身
    # 如果base_dir_path是相对路径，它会相对于当前工作目录解析。

    if not os.path.isabs(base_dir_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir_path = os.path.join(script_dir, base_dir_path)
    
    if not os.path.isdir(base_dir_path):
        print(f"错误: 目录 '{base_dir_path}' 未找到。")
        return corpus_sentences

    print(f"开始从目录 '{base_dir_path}' 加载语料库...")
    file_count = 0
    sentence_count = 0

    for root, _, files in os.walk(base_dir_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                file_count += 1
                try:
                    # 尝试使用 UTF-8 编码打开文件。
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            stripped_line = line.strip()
                            if not stripped_line: # 跳过空行
                                continue
                            
                            if re.match(r'^\d{8}-\d{2}-\d{3}-\d{3}$', stripped_line): # 跳过ID行
                                continue
                            if stripped_line.startswith(""): # 跳过特殊标记行
                                continue

                            processed_words = _parse_people_daily_line(stripped_line)
                            
                            if processed_words: 
                                corpus_sentences.append(processed_words)
                                sentence_count +=1
                except Exception as e:
                    print(f"处理文件 {file_path} 时发生错误: {e}")
    
    print(f"语料库加载操作完成。共处理 {file_count} 个文件，提取了 {sentence_count} 个句子。")
    if sentence_count == 0 and file_count > 0:
        print(f"注意: 从 {file_count} 个文件中未能提取任何有效句子。")
    return corpus_sentences


class BigramDisambiguator:
    def __init__(self, corpus_sentences):
        """
        初始化消歧器
        :param corpus_sentences: 一个列表，其中每个元素是一个已经分好词的句子 (词列表)
        """
        self.corpus_sentences = corpus_sentences
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.total_words = 0
        self.vocab_size = 0
        self._train()

    def _train(self):
        """
        根据提供的语料库训练模型，计算unigram和bigram的频率
        """
        word_set = set()
        if not self.corpus_sentences:
            print("错误: 语料库为空，模型训练中止。")
            return

        print("开始训练模型...")
        for sentence in self.corpus_sentences:
            if not sentence: 
                continue
            
            padded_sentence = sentence

            for i in range(len(padded_sentence)):
                word = padded_sentence[i]
                self.unigram_counts[word] += 1
                word_set.add(word)
                if i > 0:
                    prev_word = padded_sentence[i-1]
                    self.bigram_counts[(prev_word, word)] += 1
        
        self.total_words = sum(self.unigram_counts.values())
        self.vocab_size = len(word_set)

        if self.total_words > 0:
            print(f"模型训练完成。词汇表大小: {self.vocab_size}, 总词数: {self.total_words}。")
        else:
            print("模型训练失败: 未能从语料库中提取有效词汇。")


    def get_log_probability(self, word_sequence):
        """
        计算给定词序列的对数概率（使用Add-One平滑）
        :param word_sequence: 一个词列表
        :return: 该序列的对数概率 (float)
        """
        if not word_sequence:
            return -float('inf') 
        
        if self.vocab_size == 0 or self.total_words == 0:
            return -float('inf')


        log_prob = 0.0
        first_word = word_sequence[0]
        # 对于非常大的词汇表，total_words + vocab_size 可能会很大，导致 prob_w1 非常小
        # 但对于 Add-One 平滑，这是预期的行为
        prob_w1_numerator = self.unigram_counts.get(first_word, 0) + 1
        prob_w1_denominator = self.total_words + self.vocab_size
        
        if prob_w1_denominator == 0: # 避免除以零
            prob_w1 = 1e-20 # 赋一个极小值
        else:
            prob_w1 = prob_w1_numerator / prob_w1_denominator
        
        log_prob += math.log(prob_w1 if prob_w1 > 0 else 1e-20)


        for i in range(1, len(word_sequence)):
            prev_word = word_sequence[i-1]
            current_word = word_sequence[i]
            
            bigram_count = self.bigram_counts.get((prev_word, current_word), 0)
            prev_word_unigram_count = self.unigram_counts.get(prev_word, 0)
            
            denominator = prev_word_unigram_count + self.vocab_size
            if denominator == 0: # 避免除以零，理论上 vocab_size > 0
                 conditional_prob = 1 / self.vocab_size if self.vocab_size > 0 else 1e-20
            else:
                conditional_prob = (bigram_count + 1) / denominator
            
            log_prob += math.log(conditional_prob if conditional_prob > 0 else 1e-20)
            
        return log_prob

    def disambiguate(self, candidate_segmentations):
        """
        从多个候选分词结果中选择最可能的一个
        :param candidate_segmentations: 候选分词结果列表
        :return: (最佳分词结果, 最佳结果的对数概率, 所有结果及概率的列表)
        """
        if not candidate_segmentations:
            return None, -float('inf'), []

        if self.vocab_size == 0 or self.total_words == 0:
            return candidate_segmentations[0] if candidate_segmentations else None, \
                   -float('inf'), \
                   [{"segmentation": seg, "log_probability": -float('inf')} for seg in candidate_segmentations]

        best_segmentation = None
        max_log_prob = -float('inf')
        results_with_probs = []

        for seg in candidate_segmentations:
            log_p = self.get_log_probability(seg)
            results_with_probs.append({"segmentation": seg, "log_probability": log_p})
            if log_p > max_log_prob:
                max_log_prob = log_p
                best_segmentation = seg
        
        if best_segmentation is None and candidate_segmentations: 
            best_segmentation = candidate_segmentations[0]

        return best_segmentation, max_log_prob, results_with_probs


# --- 主程序 ---
if __name__ == "__main__":

    corpus_base_path = os.path.join("people-2014", "train") # 这里默认语料库和程序在同一目录下

    print(f"指定语料库相对路径: '{corpus_base_path}' (将解析为相对于脚本位置或当前工作目录)")
    processed_corpus = load_corpus_from_directory(corpus_base_path)
    
    if not processed_corpus:
        print(f"错误: 未能从路径 '{corpus_base_path}' (或其解析后的绝对路径) 加载语料数据。程序将退出。")
        exit()

    disambiguator = BigramDisambiguator(processed_corpus)
    
    test_cases = [
        {
            "original_sentence": "我从小学电脑",
            "candidates": [
                ["我", "从", "小学", "电脑"],
                ["我", "从小", "学", "电脑"]
            ]
        },
        {
            "original_sentence": "他喜欢研究生物化学",
            "candidates": [
                ["他", "喜欢", "研究", "生物", "化学"],
                ["他", "喜欢", "研究", "生物化学"],
                ["他", "喜欢", "研究生物", "化学"] 
            ]
        },
        {
            "original_sentence": "在北京大学生活区喝进口红酒",
            "candidates": [
                ["在", "北京大学", "生活区", "喝", "进口", "红酒"],
                ["在", "北京", "大学", "生活区", "喝", "进口", "红酒"],
                ["在", "北京大学", "生活", "区", "喝", "进口", "红酒"]
            ]
        },
        { 
            "original_sentence": "羽毛球拍卖完了 (作业示例)",
            "candidates": [
                ["羽毛球", "拍卖", "完", "了"], 
                ["羽毛", "球", "拍卖", "完", "了"] ,
                ["羽毛球拍", "卖完", "了"]
            ]
        }
    ]

    if disambiguator.vocab_size > 0 and disambiguator.total_words > 0 :
        print("\n开始进行分词歧义消解测试...")
        for i, case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: \"{case['original_sentence']}\"")
            
            best_seg, best_log_prob, all_results = disambiguator.disambiguate(case['candidates'])
            
            print("  候选分词及对数概率:")
            for res in all_results:
                seg_str = " \\ ".join(res['segmentation'])
                log_prob_display = f"{res['log_probability']:.4f}" if res['log_probability'] > -float('inf') else "-inf"
                print(f"    - \"{seg_str}\": {log_prob_display}")
                
            if best_seg:
                best_seg_str = " \\ ".join(best_seg)
                best_log_prob_display = f"{best_log_prob:.4f}" if best_log_prob > -float('inf') else "-inf"
                print(f"  预测最佳分词: \"{best_seg_str}\" (对数概率: {best_log_prob_display})")
            else:
                print("  未能确定最佳分词结果。")
    else:
        print("\n模型未成功训练或语料库为空，无法执行分词歧义消除测试。")

