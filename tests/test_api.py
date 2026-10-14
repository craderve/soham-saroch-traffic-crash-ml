import math
import unittest

from app import app, counties, input_frame, validate


class TrafficLensTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_options_and_health(self):
        self.assertEqual(self.client.get('/api/health').status_code, 200)
        self.assertEqual(self.client.get('/api/health').json['project'], 'Soham Saroch TrafficLens')
        page = self.client.get('/').get_data(as_text=True)
        self.assertIn('<title>TrafficLens | Soham Saroch</title>', page)
        self.assertIn('https://github.com/craderve/soham-saroch-traffic-crash-ml', page)
        self.assertNotIn('Sanjeev Saroch', page)
        response = self.client.get('/api/options')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Los Angeles', response.json['counties'])
        self.assertGreater(len(response.json['counties']), 100)
        self.assertIn('performance_metrics', response.json['metadata'])

    def test_feature_schema_and_weekend(self):
        frame = input_frame(validate({'day_of_week_num': 6, 'traffic_signal': True}))
        self.assertEqual(frame.iloc[0]['IsWeekend'], 1)
        self.assertEqual(frame.iloc[0]['Traffic_Signal'], 1)
        self.assertEqual(len(frame.columns), 33)

    def test_invalid_inputs(self):
        for payload in [None, [], {'temperature': 999}, {'temperature': 'NaN'},
                        {'humidity': True}, {'hour': 1.5}, {'state': 'ZZ'},
                        {'county': '<script>'}, {'traffic_signal': 'false'}, {'month': None}]:
            with self.subTest(payload=payload):
                response = self.client.post('/api/predict', json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.json)

    def test_oversized_payload(self):
        response = self.client.post('/api/predict', json={'padding': 'x' * 20000})
        self.assertEqual(response.status_code, 413)

    def test_real_model_predictions(self):
        for payload in [{}, {'temperature': 35, 'humidity': 95, 'visibility': 2,
                             'precipitation': 0.5, 'weather_condition': 'Rain',
                             'hour': 19, 'day_of_week_num': 5, 'traffic_signal': True}]:
            with self.subTest(payload=payload):
                response = self.client.post('/api/predict', json=payload)
                self.assertEqual(response.status_code, 200, response.json)
                result = response.json
                self.assertIn(result['severity'], [1, 2, 3, 4])
                self.assertTrue(-90 <= result['latitude'] <= 90)
                self.assertTrue(-180 <= result['longitude'] <= 180)
                self.assertEqual(len(result['probabilities']), 4)
                self.assertTrue(math.isclose(sum(result['probabilities']), 1, abs_tol=1e-5))
                self.assertTrue(all(0 <= p <= 1 for p in result['probabilities']))


if __name__ == '__main__':
    unittest.main()
