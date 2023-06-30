import data
import model
import streamlit as st
import matplotlib.pyplot as plt


url = r'credit_scoring.csv'
path = r'\Model'
zn = model.Model(url=url, path_model=path)

selectbox = st.sidebar.selectbox(
    "Скоринг кредита",
    ("Данные", "Модель"))

# Using "with" notation
with st.sidebar:
    if selectbox == 'Данные':
        head = st.checkbox(
            label='Исходные данные',
            value=True
        )
        get_nan = st.checkbox(
            label = 'Пропущенные значения',
            value = False
        )
        get_cor = st.checkbox(
            label = 'Кореляция данных',
            value = False
        )
        get_stat= st.checkbox(
            label = 'Статистические данные по колонкам',
            value = False 
        )


if selectbox == 'Модель':
    
    st.header('Заполните все данные')
    d1 = st.slider("Возраст", min_value=1, max_value=100, value=20,
                            step=1)
    d2 = st.number_input('Общий баланс средств', key='d1')
    d3 = st.selectbox("Возрастная группа", ("a", "b", "c", "d","e"))
    d4 = st.number_input('Просрочка в 30-59 дней за 2 года', key='d2')
    d5 = st.number_input('Ежемесячные расходы', key='d3')
    d6 = st.number_input('Ежемесячный доход', key='d4')
    d7 = st.number_input('Кол-во открытых кредитов', key='d5')
    d8 = st.number_input('Сколько раз наблюдалась просрочка (90 и более дней)', key='d6')
    d9 = st.number_input('Кол-во задержаных платежей на 60-89 дней за 2 года', key='d7')
    d10 = st.number_input('Кол-во иждивенцев на попечении', key='d8')
    d11 = st.selectbox('Закодированное количество кредиов', ("A", "B", "C", "D","E"))


    
    if st.button('Предсказать'):
        with open('pred.csv', 'w') as file:
            file.write('age,RevolvingUtilizationOfUnsecuredLines,GroupAge,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents,RealEstateLoansOrLines\n')
            file.write(f'{d1},{d2},{d3},{d4},{d5},{d6},{d7},{d8},{d9},{d10},{d11}')
        st.write("Предсказание")
        out = int(zn.load_predict())
        st.write(f"данные будут отнесены к группе - {out}")


if selectbox == 'Данные':
    if head:
        st.header('Исходные данные')
        st.write(zn.data)

    if get_nan:
        st.header('Пропущенные значения по столбцам в процентном соотношении')
        st.write(data.pd.DataFrame(list(zn.get_nan().items()), columns = ['Columns', 'NaN %']))

    if get_cor:
        st.header('Кореляция данных')
        fig, ax = plt.subplots()  
        columns = [i for i in zn.data.columns if zn.data[i].dtype!=object]
        ax = data.sns.heatmap(zn.data[columns].corr(), 
                            annot = True, 
                            vmin=-1, 
                            vmax=1, 
                            center= 0, 
                            cmap= 'coolwarm', 
                            linewidths=3, 
                            linecolor='black')
        st.pyplot(fig)


    if get_stat:
        st.header('Статистика по столбцам')
        zn.stat()
        
        for i in zn.stats_col.keys():
            st.subheader(f'Статистика по столбцу {i}')
            st.subheader(zn.inf[i])
            st.write(data.pd.DataFrame(list(zn.stats_col[i].items()), columns = ['Название', 'Данные']))
            fig, ax = plt.subplots()
            
            ax = data.sns.boxplot(data=zn.data[i])

            ax.set_xticklabels([i])
            ax.set_ylabel('Value')
            ax.set_title(f'{i} - {zn.inf[i]}')
            st.pyplot(fig)

   