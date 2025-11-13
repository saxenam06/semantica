# Real Data Sources Used in Cookbook

This document lists all real data sources, APIs, feeds, and database patterns used throughout the cookbook notebooks.

## Cybersecurity Domain

### Threat Intelligence Feeds
- **CISA Security Advisories**: `https://www.cisa.gov/news.xml`
- **US-CERT Alerts**: `https://www.us-cert.gov/ncas/alerts.xml`
- **Security Week**: `https://feeds.feedburner.com/SecurityWeek`
- **Dark Reading**: `https://www.darkreading.com/rss.xml`
- **Krebs on Security**: `https://krebsonsecurity.com/feed/`

### Threat Intelligence APIs
- **MITRE ATT&CK Framework**: `https://api.github.com/repos/mitre/cti`
- **VirusTotal API**: `https://www.virustotal.com/vtapi/v2/domain/report` (requires API key)
- **Shodan API**: `https://api.shodan.io/shodan/host/search` (requires API key)
- **CISA KEV Catalog**: `https://www.cisa.gov/known-exploited-vulnerabilities-catalog`
- **NIST NVD**: `https://nvd.nist.gov/vuln/search`

### CVE and Vulnerability Sources
- **NVD Recent CVEs (JSON)**: `https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-recent.json.zip`
- **NVD Recent CVEs (XML)**: `https://nvd.nist.gov/feeds/xml/cve/2.0/nvdcve-2.0-recent.xml.zip`
- **CVE MITRE All Items**: `https://cve.mitre.org/data/downloads/allitems.csv`
- **CISA KEV Catalog**: `https://www.cisa.gov/known-exploited-vulnerabilities-catalog/json`
- **NVD CVE API v2.0**: `https://services.nvd.nist.gov/rest/json/cves/2.0`
- **CVE Project GitHub**: `https://api.github.com/repos/CVEProject/cvelist`
- **CVE Search API**: `https://cve.circl.lu/api/last`

### Database Patterns (Cybersecurity)
```sql
-- Threat Intelligence Database
postgresql://user:password@localhost:5432/threat_intel_db
SELECT ioc, ioc_type, timestamp, severity, source 
FROM threat_indicators 
WHERE timestamp > NOW() - INTERVAL '7 days'

-- Security Logs Database
postgresql://user:password@localhost:5432/security_logs_db
SELECT * FROM security_events 
WHERE timestamp > NOW() - INTERVAL '1 hour' 
ORDER BY timestamp DESC LIMIT 1000

-- Vulnerability Database
postgresql://user:password@localhost:5432/vulnerability_db
SELECT cve_id, description, severity, published_date, affected_products 
FROM vulnerabilities 
WHERE published_date > NOW() - INTERVAL '30 days' 
ORDER BY published_date DESC
```

### Streaming Sources (Cybersecurity)
```python
# Kafka Configuration
{
    "type": "kafka",
    "topic": "security_logs",
    "bootstrap_servers": ["localhost:9092"],
    "consumer_config": {"group_id": "semantica_security_monitor"}
}

# RabbitMQ Configuration
{
    "type": "rabbitmq",
    "queue": "security_events",
    "connection_url": "amqp://user:password@localhost:5672/"
}
```

## Finance Domain

### Financial News Feeds
- **Reuters Business**: `https://feeds.reuters.com/reuters/businessNews`
- **Reuters Top News**: `https://feeds.reuters.com/reuters/topNews`
- **CNN Money**: `https://rss.cnn.com/rss/money_latest.rss`
- **Bloomberg Markets**: `https://feeds.bloomberg.com/markets/news.rss`
- **Financial Times**: `https://www.ft.com/?format=rss`

### Financial APIs
- **Polygon.io**: `https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31` (requires API key)
- **Alpha Vantage**: `https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&apikey=demo`
- **Yahoo Finance API**: `https://api.github.com/repos/ranaroussi/yfinance`

### Database Patterns (Finance)
```sql
-- Market Data Database
postgresql://user:password@localhost:5432/market_data_db
SELECT symbol, price, volume, timestamp 
FROM market_data 
WHERE timestamp > NOW() - INTERVAL '1 day' 
ORDER BY timestamp DESC

-- Transactions Database
postgresql://user:password@localhost:5432/transactions_db
SELECT transaction_id, user_id, amount, merchant, location, timestamp, device 
FROM transactions 
WHERE timestamp > NOW() - INTERVAL '24 hours' 
ORDER BY timestamp DESC LIMIT 10000
```

### Streaming Sources (Finance)
```python
# Kafka Configuration
{
    "type": "kafka",
    "topic": "transactions",
    "bootstrap_servers": ["localhost:9092"],
    "consumer_config": {"group_id": "fraud_detection"}
}

# RabbitMQ Configuration
{
    "type": "rabbitmq",
    "queue": "payment_events",
    "connection_url": "amqp://user:password@localhost:5672/"
}
```

## Healthcare Domain

### Medical News Feeds
- **CDC Health Alerts**: `https://www.cdc.gov/rss.xml`
- **WHO News**: `https://www.who.int/rss-feeds/news-english.xml`

### Healthcare APIs
- **Logica Health FHIR API**: `https://api.logicahealth.org/fhir/R4/Patient`
- **HAPI FHIR Server**: `https://hapi.fhir.org/baseR4/Patient`

### Database Patterns (Healthcare)
```sql
-- Patient Records Database (HIPAA Compliant)
postgresql://user:password@localhost:5432/patient_records_db
SELECT patient_id, visit_date, diagnosis, medication, doctor 
FROM patient_visits 
WHERE visit_date > CURRENT_DATE - INTERVAL '1 year' 
ORDER BY visit_date DESC
```

## General Public APIs

### GitHub APIs
- **MITRE ATT&CK**: `https://api.github.com/repos/mitre/cti`
- **CVE Project**: `https://api.github.com/repos/CVEProject/cvelist`
- **Yahoo Finance**: `https://api.github.com/repos/ranaroussi/yfinance`

### Public Data Sources
- **JSONPlaceholder**: `https://jsonplaceholder.typicode.com` (for testing)
- **Public APIs**: Various endpoints for demonstration

## Usage Notes

1. **API Keys**: Some APIs (VirusTotal, Shodan, Polygon.io) require API keys. Configure these in your environment or config files.

2. **Rate Limits**: Many public APIs have rate limits. Implement appropriate delays and caching.

3. **Database Connections**: Replace placeholder credentials with actual database credentials. Use environment variables or secure config files.

4. **Streaming**: Configure Kafka/RabbitMQ with actual connection details for production use.

5. **Error Handling**: All notebooks include try-except blocks to handle network errors gracefully.

6. **Feed Formats**: RSS/Atom feeds are automatically parsed by `FeedIngestor`.

7. **Data Formats**: 
   - JSON: Parsed with `JSONParser`
   - XML: Parsed with `XMLParser`
   - CSV: Parsed with `CSVParser`
   - Database: Queried with `DBIngestor`

## Best Practices

1. **Use Lists**: Pass multiple feed URLs in lists for batch processing
2. **Error Handling**: Always wrap ingestion calls in try-except blocks
3. **Logging**: Use print statements with ✓ and ⚠ symbols for clear feedback
4. **Configuration**: Store sensitive credentials in environment variables
5. **Caching**: Implement caching for frequently accessed feeds
6. **Rate Limiting**: Respect API rate limits and implement delays

## Example Usage Pattern

```python
# Real feed URLs in a list
feeds = [
    "https://www.cisa.gov/news.xml",
    "https://www.us-cert.gov/ncas/alerts.xml"
]

# Process all feeds
feed_data_list = []
for feed_url in feeds:
    try:
        feed_data = feed_ingestor.ingest_feed(feed_url)
        if feed_data:
            feed_data_list.append(feed_data)
            print(f"✓ Ingested: {feed_data.title}")
    except Exception as e:
        print(f"⚠ Error: {str(e)[:100]}")
```

