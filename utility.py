import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
from gensim import corpora, models, similarities
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import regex
import emoji
 

def process_special_word(text):
    #có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()
def process_postag_thesea(text):
    new_document = []
    for sentence in sent_tokenize(text):
        # Remove periods as per original logic (consider if this is necessary for your use case)
        sentence = sentence.replace('.', '')
        # Tokenize
        tokens = word_tokenize(sentence, format="text")
        
        
        # Append the processed sentence to the new document
        new_document.append(tokens)
    
    # Join sentences and remove excess whitespace
    new_document = ' '.join(new_document)
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    
    return new_document   
def remove_stopword(text):
    # Load StopWords
    file = open('data/vietnamese-stopwords.txt','r', encoding='utf8')
    stopwords = file.read().split('\n')
    file.close()
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document
def remove_words(text):
    text = text.lower()
    remove_word_list = ['danh mục','vui lòng', 'liên hệ','địa chỉ','inbox','sản xuất','đặt hàng','tư vấn','', 'shopee','thời trang nam','thời trang', 'phong cách', 'dáng kiểu', 'kiểu dáng','kho hàng', 'mô tả', 'thông tin', 'sản phẩm', 'xuất xứ', 'chất liệu',  'gửi từ', 'thương hiệu', 'phong cách',  'mô tả ','dành cho','họa tiết','màu sắc','giới tính', 'giao hàng', 'giỏ hàng', 'shop', ]
    # Split the text into lines
    lines = text.split('\n')
    updated_lines = []
    for line in lines:
        # Remove the words from the line
        for word in remove_word_list:
            line = line.replace(word, '')
        updated_lines.append(line)
    # Join the filtered lines back into a single string with newline characters
    cleaned_text = '\n'.join(updated_lines)
    return cleaned_text
def clean_text(text):
    text = str(text).lower()
    text = remove_words(text)
    #Loại bỏ ký tự đặc biệt
    text = re.sub('[\.\:\,\-\-\-\+\d\!\%\...\.\"\*\>\<\^\&\/\[\]\(\)\=\~●•#_—]',' ',text)
    #Loại bỏ emoji
    text = emoji.replace_emoji(text)
    text = remove_words(text)
    #Loại bỏ các từ không cần thiết
    text = re.sub('\ss\s|\sm\s|\sl\s|\sxl|xxl|xxx|xxxxl|2xl|3xl|4xl|size|\smm\s|\scm\s|\sm\s|\sg\s|\skg\s',' ',text)
    #Loại bỏ khoảng trắng thừa
    text = re.sub('\s+',' ',text)
    text = process_special_word(text)
    
    text = process_postag_thesea(text)
    text = remove_stopword(text)
    return text 

def convert_id2dataframeproduct(id_list, data_products):
    result = data_products[data_products['product_id'].isin(id_list)]
    return result
def recommend_products_collaborativefiltering(user_id, data_ratings, data_products, recommendation_model, n=5 ):
    data = data_ratings
    list_products = []
    # nếu user_id không có trong data thì sẽ recommend sản phẩm dựa trên số lượng rating và trung bình rating
    if user_id not in data['user_id'].unique() or user_id is None:
        product_ratings = data.groupby('product_id').agg(
            average_rating=('rating', 'mean'),
            number_of_ratings=('rating', 'count')
        ).reset_index()
        product_ratings = product_ratings[(product_ratings['number_of_ratings'] > 100 )& (product_ratings['average_rating'] > 4.5)].sort_values(by='average_rating', ascending=False)
        for i in range(len(product_ratings)):
            list_products.append((int(product_ratings.iloc[i]['product_id']), product_ratings.iloc[i]['average_rating']))
    else:
    # nếu user_id có trong data thì sẽ recommend sản phẩm dựa trên dự đoán rating của user_id cho sản phẩm
        for product_id in data['product_id'].unique():
            list_products.append((product_id, recommendation_model.predict(user_id, product_id).est))
        list_products = sorted(list_products, key=lambda x: x[1], reverse=True)
    list_products = [i[0] for i in list_products][:n]

    # Đổi list id thành dataframe chứa thông tin products
    list_products = convert_id2dataframeproduct(list_products, data_products)[['product_id', 'product_name', 'price', 'description']]
    return list_products
def recommendation_gensim(view_product, data_products, number_of_similar_product = 5): 
    # load model 
    tfidf = models.TfidfModel.load('models/project2/gensim/tfidf.model')
    index = similarities.SparseMatrixSimilarity.load('models/project2/gensim/index.model')
    dictionary = corpora.Dictionary.load('models/project2/gensim/dictionary.model')

    # clean text truyền vào
    view_product = clean_text(view_product).split()
    # chuyển text thành vector
    kw_vector = dictionary.doc2bow(view_product)
    sim = index[tfidf[kw_vector]]
    list_id = [] 
    list_score = []
    for i in range(len(sim)):
        if sim[i] > 0:
            list_id.append(i)
            list_score.append(sim[i])

    # tạo dataframe chứa id và score
    df_result = pd.DataFrame({'id':list_id, 'score':list_score})

    # sắp xếp theo score lấy x sản phẩm đề xuất
    top_score = df_result.sort_values(by='score', ascending=False).head(number_of_similar_product+1)
    id_to_list = top_score['id'][1:].tolist()
    score_to_list = top_score['score'][1:].tolist()

    products_result = data_products[data_products.index.isin(id_to_list)]
    result = products_result[['product_id', 'product_name', 'price', 'description']]

    result = result.assign(gensim_similarity=[score_to_list[id_to_list.index(i)] for i in result.index]).sort_values(by='gensim_similarity', ascending=False)

    return result
def recommendation_cosin(view_product_name, data_products,  number_of_similar_product = 5):
    # load model 
    tf = pickle.load(open('models/project2/cosine/tf.pkl', 'rb'))
    #load tfidf_matrix 
    tfidf_matrix = tf.transform(data_products.all_text)
    # transform view_product_name to tfidf vector
    view_product_name = clean_text(view_product_name)
    view_product_tf = tf.transform([view_product_name])
    # calculate cosine similarity for view_product with all products
    cosine_similarities = cosine_similarity(view_product_tf, tfidf_matrix)
    data_products['cosine_similarity'] = cosine_similarities[0]
    result = data_products[['product_id','product_name','price','description', 'cosine_similarity' ]].sort_values(by='cosine_similarity', ascending=False).head(number_of_similar_product)
    return result
def recommend_products(data_ratings, data_products, surprise_model, str_search = None, user_id = None,  number_of_recommen = 5): 
    df_recommend_products = pd.DataFrame()
    # Trường hợp không có user_id và không có lịch sử tìm kiếm
    if user_id is None and str_search is None:
        df_recommend_products = recommend_products_collaborativefiltering(user_id,data_ratings,data_products,surprise_model, number_of_recommen)
    # Trường hợp không có user_id và có lịch sử tìm kiếm
    elif user_id is None and str_search is not None:
        df_recommend_cosin = recommendation_cosin(str_search,data_products, number_of_recommen)
        df_recommend_gensim = recommendation_gensim(str_search, data_products, number_of_recommen)
        # Ta sẽ join 2 dataframe df_recommend_cosin và df_recommend_gensim theo product_id, product_name, price, description
        df_recommend_join  = pd.merge(df_recommend_cosin, df_recommend_gensim, on=['product_id', 'product_name', 'price', 'description'], how='outer')
        df_recommend_join['max_cosin_gensim'] = df_recommend_join[['cosine_similarity', 'gensim_similarity']].max(axis=1)
        df_recommend_join = df_recommend_join.sort_values('max_cosin_gensim', ascending=False)
        # lấy x sản phẩm đầu tiên
        df_recommend_products = df_recommend_join.head(number_of_recommen)
    # Trường hợp có user_id và có lịch sử tìm kiếm
    elif user_id is not None and str_search is not None: 
        df_recommend_products_colla = recommend_products_collaborativefiltering(user_id,data_ratings,data_products,surprise_model, number_of_recommen)
        df_recommend_cosin = recommendation_cosin(str_search,data_products, number_of_recommen)
        df_recommend_gensim = recommendation_gensim(str_search, data_products, number_of_recommen)
        df_recommend_gensim.drop(columns=['gensim_similarity'], inplace=True)
        df_recommend_cosin.drop(columns=['cosine_similarity'], inplace=True)
        # Ta sẽ join 3 dataframe df_recommend_products_colla, df_recommend_cosin và df_recommend_gensim theo product_id, product_name, price, description
        df_recommend_join  = pd.merge(df_recommend_products_colla, df_recommend_cosin, on=['product_id', 'product_name', 'price', 'description'], how='outer')
        df_recommend_join  = pd.merge(df_recommend_join, df_recommend_gensim, on=['product_id', 'product_name', 'price', 'description'], how='outer')
        # Order theo price và lấy x sản phẩm đầu tiên
        df_recommend_products = df_recommend_join.sort_values('price', ascending=True).head(number_of_recommen)
    # Trường hợp có user_id và không có lịch sử tìm kiếm
    elif user_id is not None and str_search is None:
        df_recommend_products = recommend_products_collaborativefiltering(user_id,data_ratings,data_products,surprise_model, number_of_recommen)
    return df_recommend_products[['product_id', 'product_name', 'price', 'description']]