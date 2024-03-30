import streamlit as st 
import pandas as pd
from utility import *
from joblib import load
import io

@st.cache_data
def load_SVD_model(num_parts, prefix='models/project2/surprise/recommendation_CollaborativeFiltering_model_part_'):
    full_model_bytes = b''

    # Concatenate each part
    for i in range(num_parts):
        with open(f'{prefix}{i}.joblib', 'rb') as f:
            full_model_bytes += f.read()

    # Load the model directly from the bytes in memory
    model = load(io.BytesIO(full_model_bytes))
    return model

@st.cache_data
def load_data_products():
    df_products = pd.read_csv('data/project2/Products_ThoiTrangNam_cleaned_part1.csv')
    df_products_2 = pd.read_csv('data/project2/Products_ThoiTrangNam_cleaned_part2.csv')
    df_products = pd.concat([df_products, df_products_2], axis=0)
    return df_products
@st.cache_data
def load_data_ratings():
    df_ratings = pd.read_csv('data/project2/Products_ThoiTrangNam_rating_cleaned.csv')
    return df_ratings

surprise_model = load_SVD_model(2)
df_products = load_data_products()
df_ratings = load_data_ratings()    
st.image('data/project2/images/topic.png', caption='Shoppe')
st.title("Đồ Án Tốt Nghiệp Data Science - Machine Learning")
st.write("""### Thành viên nhóm:
- Huỳnh Văn Tài
- Trần Thế Lâm""") 
menu = ["Home", "Build Project" ,"Recommendation System Prediction"]
choice = st.sidebar.selectbox('Danh mục', menu)
if choice == 'Home': 
    
    st.write("""# Đề tài: Xây dựng hệ thống đề xuất sản phẩm cho khách hàng cho sàn thương mại điện tử Shoppe""")   
    st.write("""### Mục tiêu:
    - Xây dựng hệ thống đề xuất sản phẩm cho khách hàng
    - Dựa vào lịch sử tìm kiếm, rating, sản phẩm đang xem để đề xuất sản phẩm
    - Sử dụng các phương pháp: Collaborative Filtering, Content-based Cosine Filtering, Content-based Gensim""")
    st.write("""### Dữ liệu:
    - Dữ liệu gồm 2 bảng: 
        - Bảng Products: chứa thông tin sản phẩm
        - Bảng Ratings: chứa thông tin rating của người dùng đối với sản phẩm""")
    st.write("""### Công nghệ:
    - Ngôn ngữ lập trình: Python
    - Thư viện: Pandas, Numpy, Matplotlib, Seaborn, Gensim, Scikit-learn, Streamlit""")
    st.write("""### Kết quả:
    - Đề xuất sản phẩm cho khách hàng vãng lai mới
    - Đề xuất sản phẩm cho khách hàng dựa vào lịch sử tìm kiếm
    - Đề xuất sản phẩm cho khách hàng dựa vào rating
    - Đề xuất sản phẩm cho khách hàng dựa vào sản phẩm đang xem""")
elif choice == 'Build Project':
    st.write("""# Build Project""")
    st.write("""### 1. Làm sạch dữ liệu""")
    st.write('#### Dữ liệu Ratings')
    st.write("""<class 'pandas.core.frame.DataFrame'>  
        RangeIndex: 1024482 entries, 0 to 1024481  
        Data columns (total 4 columns):  
        #   Column      Non-Null Count    Dtype   
        ---  ------      --------------    -----   
        0   product_id  1024482 non-null  int64   
        1   user_id     1024482 non-null  int64   
        2   user        1024482 non-null  object  
        3   rating      1024482 non-null  int64   
        dtypes: int64(3), object(1)  """)
    st.write('#### Thống kê dữ liệu Ratings')
    st.image('data/project2/images/ThongKeRatings.png', caption='Thống kê Ratings')
    st.write('#### Lọc các user spam (user đánh giá cùng 1 sản phẩm và cùng rating lớn hơn 3)')
    st.write("Các user spam: 'Người dùng Shopee', 't*****1', 't*****2', 't*****3', 'n*****1', 't*****5', 't*****9', 'n*****3', 't*****7', 't*****0', 'n*****2', 't*****6', 'n*****9', 't*****8', 't*****4', 'h*****1', 'h*****2', 'h*****3', 'n*****4', 'n*****0', 'h*****4', 'n*****8', 'n*****7', 'n*****6', 'h*****9', 'n*****5', 'h*****5', 'h*****n', 'h*****8', 't*****n', 'h*****6', 'm*****9', 'h*****7', 'h*****0', 'l*****3', 'ngokinh3', 'm*****1', 'l*****2', 't*****_'")
    st.write('#### Dữ liệu Products')
    st.write("""<class 'pandas.core.frame.DataFrame'>  
            RangeIndex: 49663 entries, 0 to 49662  
            Data columns (total 9 columns):  
            #   Column        Non-Null Count  Dtype    
            ---  ------        --------------  -----    
            0   product_id    49663 non-null  int64    
            1   product_name  49663 non-null  object   
            2   category      49663 non-null  object   
            3   sub_category  49663 non-null  object   
            4   link          49663 non-null  object   
            5   image         36443 non-null  object   
            6   price         49663 non-null  float64  
            7   rating        49663 non-null  float64  
            8   description   48700 non-null  object   
            dtypes: float64(2), int64(1), object(6)  """)
    st.write('#### Wordcloud của dữ liệu Products')
    st.image('data/project2/images/WordCloud_dfProducts.png', caption='Wordcloud Products')
    st.write("""### 2. Xây dựng mô hình Collaborative Filtering""")
    st.write('#### ALS Model')
    st.write('Data ratings')
    st.write("""+----------+-------+------------------+------+  
        |product_id|user_id|              user|rating|  
        +----------+-------+------------------+------+  
        |       190|      1|      karmakyun2nd|     5|  
        |       190|      2|  tranquangvinh_vv|     5|  
        |       190|      3|nguyenquoctoan2005|     5|  
        |       190|      4|    nguyenthuyhavi|     5|  
        |       190|      5|      luonganh5595|     5|  
        +----------+-------+------------------+------+""")
    st.write('Root Mean Square Error (RMSE) của mô hình ALS')
    st.image('data/project2/images/ALS_RMSE.png', caption='ALS RMSE')
    st.write('#### SVD Model')
    st.write('Root Mean Square Error (RMSE) của mô hình SVD')
    st.image('data/project2/images/SVD_RMSE.png', caption='SVD RMSE')
    st.write("""### Nhận xét:  
    - Mô hình SVD cho kết quả tốt hơn mô hình ALS với RMSE ~0.85
    - Lựa chọn Surprise SVD model để xây dựng hệ thống đề xuất sản phẩm theo rating""")
    st.write("""### 3. Xây dựng mô hình Content-based Cosine Filtering""")
    st.write('#### Mô hình Content-based Cosine Filtering')
    st.write('#### Thực hiện tìm kiếm sản phẩm cho từ khóa "đồ thể thao"')
    st.image('data/project2/images/gensim_predict_result.png', caption='Gensim result')
    st.image('data/project2/images/gensim_predict_wordcloud.png', caption='Content-based Cosine Filtering')
    st.write('#### Mô hình Content-based Cosine Filtering')
    st.write('#### Thực hiện tìm kiếm sản phẩm cho từ khóa "đồ thể thao"')
    st.image('data/project2/images/cosine_predict_result.png', caption='Cosine result')
    st.image('data/project2/images/cosine_predict_wordcloud.png', caption='Content-based Cosine Filtering')

    st.write("""### Nhận xét: 
    - Cả 2 Mô hình đều cho được kết quả tương đối khả quan và phù hợp với từ khóa tìm kiếm => kết hợp cả 2 model để đề xuất sản phẩm cho khách hàng dựa trên score của 2 model""")






elif choice == 'Recommendation System Prediction':
    st.write("""# Recommendation System""")
    
    df_users = df_ratings[['user_id','user']].drop_duplicates().set_index('user_id')
    st.write('### Dữ liệu Ratings demo')
    st.write(df_ratings.sample(5))
    st.write('### Dữ liệu Products demo')
    st.write(df_products.sample(5))
    st.write('### Dữ liệu Users demo')
    st.write(df_users.sample(5))

    st.write('### 1.Đề xuất cho khách hàng vãng lai mới')
    st.write('##### Đưa ra 10 đề xuất cho với các sản phẩm có số lượng rating nhiều nhất và trung bình rating trên 4.5')
    list_products1 = recommend_products(df_ratings, df_products,surprise_model, 5).sort_values(by="price").set_index('product_id')
    st.write(list_products1)

    st.write('### 2. Đề xuất tìm kiếm sản phẩm cho khách hàng')
    input2 = st.text_input('Từ khóa tìm kiếm: ')
    button2_timkiem =st.button('Tìm kiếm')
    if button2_timkiem:
        if input2:
            list_products2 = recommendation_cosin(input2, df_products, 5).set_index('product_id') [['product_name', 'price', 'description']]
            st.write(list_products2)

    st.write('### 3. Gợi ý cho khách hàng có lịch sử tìm kiếm')
    st.session_state.user_history = ['Bộ nỉ nam năng động tay dài', 'Bộ nỉ dày nam nữ mặc siêu ấm, set nỉ nam có mũ, quần áo thể thao thu đông cực ấm', ]
    st.write('Lịch sử tìm kiếm của khách hàng: ')
    search_string =  ', '.join(st.session_state.user_history)
    st.write(search_string)
    input3 = st.text_input('Thêm lịch sử tìm kiếm: ')
    button3 = st.button('Thêm')
    if button3:
        if input3:
            st.session_state.user_history.append(input3)
        st.session_state.user_history = st.session_state.user_history[-5:]
        search_string =  ', '.join(st.session_state.user_history)
        st.write('Lịch sử tìm kiếm của khách hàng:')
        st.write(search_string)
        list_products3 = recommendation_cosin(search_string, df_products, 5).set_index('product_id')[['product_name', 'price', 'description']]
        st.write(list_products3)
    
    
    st.write('### 4. Đề xuất cho khách hàng dựa vào rating')
    input4 = st.text_input('Nhập Mã số khách hàng: ')
    button4 = st.button('Random user')
    
    if input4: 
        input4 = int(input4)
        st.write('Đề xuất sản phẩm cho khách hàng id: ', input4)
        list_products4 = recommend_products_collaborativefiltering(int(input4),df_ratings, df_products, surprise_model, 5).set_index('product_id')
        st.write(list_products4)
    if button4:
        user_id4 = df_ratings.sample(1)['user_id'].values[0]
        st.write('Đề xuất sản phẩm cho khách hàng id: ', user_id4)
        list_products4 = recommend_products_collaborativefiltering(user_id4,df_ratings, df_products, surprise_model, 5).set_index('product_id')
        st.write(list_products4)


    st.write('### 5. Đề xuất cho khách hàng dựa vào sản phẩm đang xem')
    product_id5 = st.text_input('Nhập mã sản phẩm: ', key='product_id')
    button5 = st.button('Random product')
    
        
    if product_id5:
        product_id5 = int(product_id5)
        product_str = df_products[df_products['product_id'] == product_id5]['product_name'].values[0] + ' ' + df_products[df_products['product_id'] == product_id5]['description'].values[0]
        st.write('Sản phẩm đang xem: ', df_products[df_products['product_id'] == product_id5]['product_name'].values[0])
        
        st.write("Sản phẩm tương tự")
        list_products5 = recommendation_cosin(product_str, df_products, 6).set_index('product_id')[['product_name', 'price', 'description']][-5:]
        st.write(list_products5)
    if button5:
        product_id5 = df_products.sample(1)['product_id'].values[0]
        st.write('Sản phẩm đang xem: ', df_products[df_products['product_id'] == product_id5]['product_name'].values[0])
        product_str = df_products[df_products['product_id'] == product_id5]['product_name'].values[0] + ' ' + df_products[df_products['product_id'] == product_id5]['description'].values[0]
        st.write("Sản phẩm tương tự")
        list_products5 = recommendation_cosin(product_str, df_products, 6).set_index('product_id')[['product_name', 'price', 'description']][-5:]
        st.write(list_products5)