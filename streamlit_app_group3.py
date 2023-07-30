import streamlit as st
st.title('SpeedyBytes 🚚')
st.image('speedybytes icon.jpg', caption='Image Caption', use_column_width=True)
tab1,tab2,tab3,tab4,tab5 = st.tabs(["tab1", "One year revenue forecast", "Optimal Shift Timing Recommendation",'tab4','tab5'])
with tab1: #ayrton
    st.header('omg tab1 works yay')


with tab2: #minh
    print('gelo')

with tab3: #javier
    st.header('Optimal Shift Timing Recommendation')
    st.image('speedybytes icon.jpg', caption='Image Caption', use_column_width=True)



with tab4: #vibu
    print('vibu')

with tab5: #nathan
    print('nathan')
    
