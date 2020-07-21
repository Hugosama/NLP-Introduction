<!-- wp:heading -->
<h2 id="%3Cstrong%3E%E8%B5%9B%E9%A2%98%E7%90%86%E8%A7%A3%3C%2Fstrong%3E"><strong>赛题理解</strong></h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>赛题名称：零基础入门NLP之新闻文本分类</li><li>赛题目标：通过这道赛题可以引导大家走入自然语言处理的世界，带大家接触NLP的预处理、模型构建和模型训练等知识点。</li><li>赛题任务：赛题以自然语言处理为背景，要求选手对新闻文本进行分类，这是一个典型的字符识别问题。</li></ul>
<!-- /wp:list -->

<!-- wp:heading {"level":3} -->
<h3 id="%3Cstrong%3E%E5%AD%A6%E4%B9%A0%E7%9B%AE%E6%A0%87%3C%2Fstrong%3E"><strong>学习目标</strong></h3>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>理解赛题背景与赛题数据</li><li>完成赛题报名和数据下载，理解赛题的解题思路</li></ul>
<!-- /wp:list -->

<!-- wp:heading {"level":3} -->
<h3 id="%3Cstrong%3E%E8%B5%9B%E9%A2%98%E6%95%B0%E6%8D%AE%3C%2Fstrong%3E"><strong>赛题数据</strong></h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>赛题以匿名处理后的新闻数据为赛题数据，数据集报名后可见并可下载。赛题数据为新闻文本，并按照字符级别进行匿名处理。整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>赛题数据由以下几个部分构成：训练集20w条样本，测试集A包括5w条样本，测试集B包括5w条样本。为了预防选手人工标注测试集的情况，我们将比赛数据的文本按照字符级别进行了匿名处理。</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 id="%3Cstrong%3E%E6%95%B0%E6%8D%AE%E6%A0%87%E7%AD%BE%3C%2Fstrong%3E"><strong>数据标签</strong></h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>处理后的赛题训练数据如下：</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/1095279501877/1594906820936_hVKPJHWvu4.jpg" alt="Image"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>在数据集中标签的对应的关系如下：{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 id="%3Cstrong%3E%E8%AF%84%E6%B5%8B%E6%8C%87%E6%A0%87%3C%2Fstrong%3E"><strong>评测指标</strong></h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>评价标准为类别<code>f1_score</code>的均值，选手提交结果与实际测试集的类别进行对比，结果越大越好。</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 id="%3Cstrong%3E%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96%3C%2Fstrong%3E"><strong>数据读取</strong></h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>使用<code>Pandas</code>库完成数据读取操作，并对赛题数据进行分析。</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 id="%3Cstrong%3E%E8%A7%A3%E9%A2%98%E6%80%9D%E8%B7%AF%3C%2Fstrong%3E"><strong>解题思路</strong></h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>赛题思路分析：赛题本质是一个文本分类问题，需要根据每句的字符进行分类。但赛题给出的数据是匿名化的，不能直接使用中文分词等操作，这个是赛题的难点。</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>因此本次赛题的难点是需要对匿名字符进行建模，进而完成文本分类的过程。由于文本数据是一种典型的非结构化数据，因此可能涉及到<code>特征提取</code>和<code>分类模型</code>两个部分。为了减低参赛难度，我们提供了一些解题思路供大家参考：</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>思路1：TF-IDF + 机器学习分类器</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>直接使用TF-IDF对文本提取特征，并使用分类器进行分类。在分类器的选择上，可以使用SVM、LR、或者XGBoost。学习过程中搜到LightGBM似乎也可以，后期可以使用一下。</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>关于TF-IDF的内容可以参考：<a href="https://blog.csdn.net/asialee_bird/article/details/81486700?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&amp;depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase">https://blog.csdn.net/asialee_bird/article/details/81486700?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&amp;depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase</a></p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>思路2：FastText</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>FastText是入门款的词向量，利用Facebook提供的FastText工具，可以快速构建出分类器。</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>简单实现：<a href="https://blog.csdn.net/yangfengling1023/article/details/86614797">https://blog.csdn.net/yangfengling1023/article/details/86614797</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>稍微深入：<a href="https://blog.csdn.net/ZJRN1027/article/details/98340304">https://blog.csdn.net/ZJRN1027/article/details/98340304</a></p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>思路3：WordVec + 深度学习分类器</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>WordVec是进阶款的词向量，并通过构建深度学习分类完成分类。深度学习分类的网络结构可以选择TextCNN、TextRNN或者BiLSTM。深度学习分类网络有点难，暂时打扰。</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>思路4：Bert词向量</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>Bert是高配款的词向量，具有强大的建模学习能力。了解Bert后发现需要GPU资源，直接打扰。</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->
