import unittest
from unittest.mock import MagicMock, patch
from semantica.triplet_store.triplet_manager import TripletManager, TripletStore
from semantica.triplet_store.query_engine import QueryEngine, QueryResult
from semantica.semantic_extract.triplet_extractor import Triplet

class TestTripletStore(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()
        self.mock_tracker = MagicMock()
        
        self.logger_patcher = patch('semantica.triplet_store.triplet_manager.get_logger', return_value=self.mock_logger)
        self.tracker_patcher = patch('semantica.triplet_store.triplet_manager.get_progress_tracker', return_value=self.mock_tracker)
        self.logger_patcher_qe = patch('semantica.triplet_store.query_engine.get_logger', return_value=self.mock_logger)
        self.tracker_patcher_qe = patch('semantica.triplet_store.query_engine.get_progress_tracker', return_value=self.mock_tracker)
        
        self.logger_patcher.start()
        self.tracker_patcher.start()
        self.logger_patcher_qe.start()
        self.tracker_patcher_qe.start()

    def tearDown(self):
        self.logger_patcher.stop()
        self.tracker_patcher.stop()
        self.logger_patcher_qe.stop()
        self.tracker_patcher_qe.stop()

    def test_triplet_manager_init(self):
        manager = TripletManager(default_store="main")
        self.assertEqual(manager.default_store_id, "main")
        self.assertEqual(manager.stores, {})

    def test_register_store(self):
        manager = TripletManager()
        store = manager.register_store("main", "blazegraph", "http://localhost:9999")
        self.assertIsInstance(store, TripletStore)
        self.assertEqual(store.store_id, "main")
        self.assertEqual(store.store_type, "blazegraph")
        self.assertEqual(store.endpoint, "http://localhost:9999")
        self.assertIn("main", manager.stores)

    @patch('semantica.triplet_store.triplet_manager.TripletManager._get_adapter')
    def test_add_triplet(self, mock_get_adapter):
        manager = TripletManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.add_triplet.return_value = {"status": "success"}
        
        triplet = Triplet(subject="s", predicate="p", object="o")
        result = manager.add_triplet(triplet, store_id="main")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["store_id"], "main")
        mock_adapter.add_triplet.assert_called_once_with(triplet)

    @patch('semantica.triplet_store.triplet_manager.TripletManager._get_adapter')
    def test_add_triplets(self, mock_get_adapter):
        manager = TripletManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.add_triplets.return_value = {"status": "success"}
        
        triplets = [
            Triplet(subject="s1", predicate="p1", object="o1"),
            Triplet(subject="s2", predicate="p2", object="o2")
        ]
        
        result = manager.add_triplets(triplets, store_id="main", batch_size=2)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["store_id"], "main")
        mock_adapter.add_triplets.assert_called()

    @patch('semantica.triplet_store.triplet_manager.TripletManager._get_adapter')
    def test_get_triplets(self, mock_get_adapter):
        manager = TripletManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        expected_triplets = [Triplet(subject="s", predicate="p", object="o")]
        mock_adapter.get_triplets.return_value = expected_triplets
        
        result = manager.get_triplets(subject="s", store_id="main")
        
        self.assertEqual(result, expected_triplets)
        mock_adapter.get_triplets.assert_called_once_with("s", None, None)

    @patch('semantica.triplet_store.triplet_manager.TripletManager._get_adapter')
    def test_delete_triplet(self, mock_get_adapter):
        manager = TripletManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.delete_triplet.return_value = {"status": "deleted"}
        
        triplet = Triplet(subject="s", predicate="p", object="o")
        result = manager.delete_triplet(triplet, store_id="main")
        
        self.assertTrue(result["success"])
        mock_adapter.delete_triplet.assert_called_once_with(triplet)

    @patch('semantica.triplet_store.triplet_manager.TripletManager._get_adapter')
    def test_update_triplet(self, mock_get_adapter):
        manager = TripletManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.delete_triplet.return_value = {"status": "deleted"}
        mock_adapter.add_triplet.return_value = {"status": "added"}
        
        old_triplet = Triplet(subject="s", predicate="p", object="o_old")
        new_triplet = Triplet(subject="s", predicate="p", object="o_new")
        
        result = manager.update_triplet(old_triplet, new_triplet, store_id="main")
        
        self.assertTrue(result["success"])
        mock_adapter.delete_triplet.assert_called_once_with(old_triplet)
        mock_adapter.add_triplet.assert_called_once_with(new_triplet)

    def test_query_engine_init(self):
        engine = QueryEngine(enable_caching=True)
        self.assertTrue(engine.enable_caching)
        self.assertEqual(engine.query_cache, {})

if __name__ == '__main__':
    unittest.main()
