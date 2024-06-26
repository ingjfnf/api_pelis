from flask import Flask
from flask_restx import Api, Resource, fields
from despliegue_peliculas import predict_genres
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='1.0',
    title='API PREDICCIÓN DE GÉNEROS DE PELÍCULAS',
    description='Esta es una API que utiliza un modelo de clasificación para predecir los géneros de películas. Puede probar cuantas veces quiera y con cualquier índice que exista en el conjunto de TESTING.'
)

ns = api.namespace('predict',
                   description='Predicción de géneros de películas')

parser = api.parser()
parser.add_argument(
    'índice',
    type=int,
    required=True,
    help='Introduzca el número del índice del conjunto de TEST que desea predecir con nuestro modelo',
    location='args'
)

resource_fields = api.model('Resource', {
    'TOP 3 de Géneros Más Probables': fields.List(fields.List(fields.Raw)),
    'Probabilidades por cada género': fields.List(fields.List(fields.Raw))
})

error_fields = api.model('Error', {
    'error': fields.String
})

@ns.route('/')
class MovieGenresApi(Resource):
    @api.doc(parser=parser)
    @api.response(200, 'Success', resource_fields)
    @api.response(404, 'Índice NO EXISTE en el conjunto de testing', error_fields)
    @api.response(500, 'Error inesperado', error_fields)
    def get(self):
        args = parser.parse_args()
        indice = args['índice']
        
        # Cargamos el conjunto de datos de prueba
        df_test = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)  

        try:
            top_genres, pred_probas = predict_genres(df_test, indice)
            
            # Convertimos las probabilidades a una lista de listas
            prob_list = [[genre, float(prob)] for genre, prob in pred_probas.iloc[0].items()]
            
            return {
                "TOP 3 de Géneros Más Probables": top_genres,
                "Probabilidades por cada género": prob_list
            }, 200
        except ValueError as e:
            return {"error": "Índice NO EXISTE en el conjunto de testing"}, 404
        except Exception as e:
            return {"error": f"Error inesperado: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
