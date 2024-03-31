from FCM import fcm
from charts import charts

# k - количество кластеров 1<j<k
# d - размерность вектора данных 1<l<d
# n - мощность выборки

def main(k):
    fcm(k)
    charts(k)

if __name__ == '__main__':
    k = 3
    main(k)


