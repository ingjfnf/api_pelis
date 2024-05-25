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
    description='Esta es una API que utiliza un modelo de clasificación para predecir los géneros de películas. Puede probar cuantas veces quiera y con cualquier ID dentro del rango del conjunto de prueba.'
)

ns = api.namespace('predict',
                   description='Predicción de géneros de películas')

parser = api.parser()
parser.add_argument(
    'ID',
    type=int,
    required=True,
    help='Introduzca el número del ID del conjunto de TEST que desea predecir con nuestro modelo',
    location='args'
)

resource_fields = api.model('Resource', {
    'Géneros Más Probables': fields.List(fields.List(fields.Raw)),
    'Probabilidades': fields.List(fields.List(fields.Raw))
})

@ns.route('/')
class MovieGenresApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        indice = args['ID']
        
        # Cargamos el conjunto de datos de prueba
        df_test = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)  

        try:
            top_genres, pred_probas = predict_genres(df_test, indice)
            
            # Convertir las probabilidades a una lista de listas
            prob_list = [[genre, prob] for genre, prob in pred_probas.to_dict().items()]
            
            return {
                "Géneros Más Probables": top_genres,
                "Probabilidades": prob_list
            }, 200
        except ValueError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            api.abort(500, f"Error inesperado: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
