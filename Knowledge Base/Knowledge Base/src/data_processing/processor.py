import json
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self):
        pass

    def extract_json_data(self, file_path: str) -> Dict:
        """Extract JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print("✅ Successfully loaded JSON data")
            return data
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return {}

    def process_json_data(self, json_data) -> List[Dict]:
        """Process JSON data to flatten structure"""
        output_items = []

        if isinstance(json_data, list):
            print(f"📋 Processing direct list format with {len(json_data)} items")
            for data_item in json_data:
                if not isinstance(data_item, dict):
                    continue

                if 'content' in data_item:
                    flat_item = {
                        'title': data_item.get('title', ''),
                        'categories': json.dumps(data_item.get('metadata', {}).get('categories', [])),
                        'tags': json.dumps(data_item.get('metadata', {}).get('tags', [])),
                        'content': data_item.get('content', '')
                    }
                    output_items.append(flat_item)

        elif isinstance(json_data, dict) and 'data' in json_data:
            print("📋 Processing n8n format with 'data' key")
            for data_item in json_data['data']:
                if not isinstance(data_item, dict):
                    continue

                flat_item = {
                    'title': data_item.get('title', ''),
                    'categories': json.dumps(data_item.get('metadata', {}).get('categories', [])),
                    'tags': json.dumps(data_item.get('metadata', {}).get('tags', [])),
                    'content': data_item.get('content', '')
                }
                output_items.append(flat_item)

        print(f"✅ Processed {len(output_items)} items")
        return output_items

    def create_documents(self, processed_data: List[Dict]) -> List[Dict]:
        """Create document objects from processed data"""
        documents = []
        
        for item in processed_data:
            if not item.get('content', '').strip():
                continue

            try:
                categories = json.loads(item['categories']) if item['categories'] else []
                tags = json.loads(item['tags']) if item['tags'] else []
            except json.JSONDecodeError:
                categories = []
                tags = []

            doc = {
                'page_content': item['content'],
                'metadata': {
                    'title': item['title'],
                    'categories': categories,
                    'tags': tags
                }
            }
            documents.append(doc)

        print(f"✅ Created {len(documents)} documents")
        return documents
