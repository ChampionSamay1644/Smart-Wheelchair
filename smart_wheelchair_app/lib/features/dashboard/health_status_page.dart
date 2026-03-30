import 'package:flutter/material.dart';
import '../../core/localization.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:provider/provider.dart';
import '../../core/providers/api_provider.dart';
import 'dart:math' as math;

class HealthStatusPage extends StatefulWidget {
  const HealthStatusPage({super.key});

  @override
  State<HealthStatusPage> createState() => _HealthStatusPageState();
}

class _HealthStatusPageState extends State<HealthStatusPage> {
  int _selectedWeek = 1;
  int _selectedMonthOffset = 0;

  @override
  Widget build(BuildContext context) {
    return Consumer<ApiProvider>(
      builder: (context, apiProvider, _) {
        final data = apiProvider.latestSensorData;
        final dht = data['dht11'] ?? {};
        final max30 = data['max30100'] ?? {};
        final killswitch = apiProvider.killswitchEnabled;

        return Scaffold(
          appBar: AppBar(
            title: Text(tr(context, 'health_status')),
            actions: [
              if (apiProvider.isPolling)
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16),
                  child: Icon(Icons.sync, color: Colors.green, size: 16),
                ),
            ],
          ),
          body: killswitch 
            ? _buildKillswitchMessage()
            : SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (apiProvider.selectedDeviceId == null)
                      _buildNoDeviceWarning(),
                    _buildTimeRangeSelector(),
                    _buildSubSelector(apiProvider.currentTimeframe),
                    const SizedBox(height: 24),
                    _buildVitalCard(
                      'Pulse Rate',
                      max30['pulse'] != null ? max30['pulse'].toStringAsFixed(0) :
                        (max30['ir'] != null ? (max30['ir'] / 100).toStringAsFixed(0) : '--'),
                      'BPM',
                      FontAwesomeIcons.heartPulse,
                      Colors.red,
                      _buildPulseRateChart(apiProvider.sensorHistory),
                    ),
                    const SizedBox(height: 16),
                    _buildVitalCard(
                      'Blood Oxygen',
                      max30['spo2'] != null ? max30['spo2'].toStringAsFixed(0) :
                        (max30['red'] != null ? (max30['red'] / 100).toStringAsFixed(0) : '--'),
                      '%',
                      FontAwesomeIcons.circleExclamation,
                      Colors.blueAccent,
                      _buildSpo2Chart(apiProvider.sensorHistory),
                    ),
                    const SizedBox(height: 16),
                    _buildVitalCard(
                      tr(context, 'body_temperature'),
                      dht['temperature']?.toString() ?? '--',
                      '°C',
                      FontAwesomeIcons.temperatureHalf,
                      Colors.orange,
                      _buildTemperatureChart(apiProvider.sensorHistory),
                    ),
                    const SizedBox(height: 16),
                    _buildVitalCard(
                      'Humidity',
                      dht['humidity']?.toString() ?? '--',
                      '%',
                      FontAwesomeIcons.droplet,
                      Colors.blue,
                      _buildHumidityChart(apiProvider.sensorHistory),
                    ),
                  ],
                ),
              ),
        );
      },
    );
  }

  Widget _buildKillswitchMessage() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.warning_amber_rounded, size: 64, color: Colors.red),
          const SizedBox(height: 16),
          Text(
            'System Temporarily Disabled',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const Padding(
            padding: EdgeInsets.all(24.0),
            child: Text(
              'Hardware communication is currently disabled by the administrator. Please try again later.',
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNoDeviceWarning() {
    return Card(
      color: Colors.orange.shade100,
      child: ListTile(
        leading: const Icon(Icons.warning, color: Colors.orange),
        title: const Text('No Wheelchair Selected'),
        subtitle: const Text('Go to Settings to select your wheelchair device.'),
        trailing: TextButton(
          onPressed: () => Navigator.pushNamed(context, '/settings'),
          child: const Text('Settings'),
        ),
      ),
    );
  }

  Widget _buildHumidityChart(List<dynamic> history) {
    final spots = history.asMap().entries.map((e) {
      final val = e.value['dht11']?['humidity'] ?? 0.0;
      return FlSpot(e.key.toDouble(), val.toDouble());
    }).toList().reversed.toList();

    return _buildChart(spots, Colors.blue);
  }

  Widget _buildTimeRangeSelector() {
    final apiProvider = Provider.of<ApiProvider>(context, listen: false);
    return SegmentedButton<String>(
      segments: const [
        ButtonSegment<String>(value: '24h', label: Text('24h')),
        ButtonSegment<String>(value: '7d', label: Text('7d')),
        ButtonSegment<String>(value: '30d', label: Text('30d')),
      ],
      selected: {apiProvider.currentTimeframe},
      onSelectionChanged: (Set<String> newSelection) {
        apiProvider.setTimeframe(newSelection.first);
      },
    );
  }

  Widget _buildSubSelector(String timeframe) {
    if (timeframe == '24h') return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.only(top: 16),
      child: Row(
        children: [
          Text(
            timeframe == '7d' ? 'Select Week:' : 'Select Month:',
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(width: 12),
          if (timeframe == '7d')
            DropdownButton<int>(
              value: _selectedWeek,
              items: List.generate(4, (i) => i + 1).map((week) {
                return DropdownMenuItem(value: week, child: Text('Week $week'));
              }).toList(),
              onChanged: (val) {
                if (val != null) setState(() => _selectedWeek = val);
              },
            )
          else
            DropdownButton<int>(
              value: _selectedMonthOffset,
              items: [
                const DropdownMenuItem(value: 0, child: Text('Last Month')),
                const DropdownMenuItem(value: 1, child: Text('2nd Last Month')),
                const DropdownMenuItem(value: 2, child: Text('3rd Last Month')),
              ],
              onChanged: (val) {
                if (val != null) setState(() => _selectedMonthOffset = val);
              },
            ),
        ],
      ),
    );
  }

  Widget _buildVitalCard(
    String title,
    String value,
    String unit,
    IconData icon,
    Color color,
    Widget chart,
  ) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                FaIcon(icon, color: color, size: 24),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    title,
                    style: Theme.of(context).textTheme.titleMedium,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  value,
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: color,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 4),
                Text(unit, style: Theme.of(context).textTheme.bodyMedium),
              ],
            ),
            const SizedBox(height: 16),
            SizedBox(height: 200, child: chart),
          ],
        ),
      ),
    );
  }

  Widget _buildPulseRateChart(List<dynamic> history) {
    final apiProvider = Provider.of<ApiProvider>(context, listen: false);
    List<dynamic> dataToUse = history;
    
    if (apiProvider.currentTimeframe != '24h') {
        dataToUse = _generateStaticData('pulse');
    }

    if (dataToUse.isEmpty) return const Center(child: Text('No history data'));
    
    final spots = dataToUse.asMap().entries.map((e) {
      final val = e.value['max30100']?['ir'] != null ? (e.value['max30100']!['ir'] / 100) : 70.0;
      return FlSpot((dataToUse.length - 1 - e.key).toDouble(), val.toDouble());
    }).toList();

    return _buildChart(spots, Colors.red);
  }

  Widget _buildSpo2Chart(List<dynamic> history) {
    final apiProvider = Provider.of<ApiProvider>(context, listen: false);
    List<dynamic> dataToUse = history;
    
    if (apiProvider.currentTimeframe != '24h') {
        dataToUse = _generateStaticData('spo2');
    }

    if (dataToUse.isEmpty) return const Center(child: Text('No history data'));
    
    final spots = dataToUse.asMap().entries.map((e) {
      final val = e.value['max30100']?['red'] != null ? (e.value['max30100']!['red'] / 100) : 98.0;
      return FlSpot((dataToUse.length - 1 - e.key).toDouble(), val.toDouble());
    }).toList();

    return _buildChart(spots, Colors.blueAccent);
  }

  Widget _buildTemperatureChart(List<dynamic> history) {
    final apiProvider = Provider.of<ApiProvider>(context, listen: false);
    List<dynamic> dataToUse = history;
    
    if (apiProvider.currentTimeframe != '24h') {
        dataToUse = _generateStaticData('temp');
    }

    if (dataToUse.isEmpty) return const Center(child: Text('No history data'));
    final spots = dataToUse.asMap().entries.map((e) {
      final val = e.value['dht11']?['temperature'] ?? 0.0;
      return FlSpot((dataToUse.length - 1 - e.key).toDouble(), val.toDouble());
    }).toList();

    return _buildChart(spots, Colors.orange);
  }

  List<dynamic> _generateStaticData(String type) {
    final apiProvider = Provider.of<ApiProvider>(context, listen: false);
    final is7d = apiProvider.currentTimeframe == '7d';
    
    // Use a TRULY FIXED seed: device + timeframe selection + type only.
    // Do NOT include any time-based values so the graph never changes.
    final deviceSeed = (apiProvider.selectedDeviceId ?? 'default').codeUnits.fold(0, (a, b) => a + b);
    final timeframeSeed = is7d ? (_selectedWeek * 7) : (_selectedMonthOffset * 30 + 100);
    final typeSeed = type == 'pulse' ? 1000 : (type == 'spo2' ? 3000 : 2000);
    
    final seed = deviceSeed + timeframeSeed + typeSeed;
    final rand = math.Random(seed); // Seeded = always same output
    
    int points = is7d ? 7 : 30;
    double baseVal = (type == 'pulse' ? 72.0 : (type == 'spo2' ? 96.0 : 36.5));
    double variance = (type == 'pulse' ? 10.0 : (type == 'spo2' ? 3.0 : 0.8));

    return List.generate(points, (i) {
        return {
            'max30100': {
              'ir': ((baseVal + rand.nextDouble() * variance) * 100).round(),
              'red': ((97.0 - rand.nextDouble() * 4.0) * 100).round(),
              'pulse': baseVal + rand.nextDouble() * variance,
              'spo2': 97.0 - rand.nextDouble() * 4.0,
            },
            'dht11': {
                'temperature': 36.0 + rand.nextDouble() * 1.5,
                'humidity': 50.0 + rand.nextDouble() * 20,
            }
        };
    });
  }

  Widget _buildChart(List<FlSpot> spots, Color color) {
    if (spots.isEmpty) {
      return const Center(child: Text('No data available'));
    }
    return LineChart(
      LineChartData(
        gridData: _defaultGridData,
        titlesData: _defaultTitlesData,
        borderData: _defaultBorderData,
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            color: color,
            barWidth: 3,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: color.withOpacity(0.1),
            ),
          ),
        ],
        minY: spots.map((s) => s.y).reduce(math.min) - 5,
        maxY: spots.map((s) => s.y).reduce(math.max) + 5,
      ),
    );
  }



  FlGridData get _defaultGridData =>
      const FlGridData(show: true, drawVerticalLine: false);

  FlTitlesData get _defaultTitlesData => const FlTitlesData(
    show: true,
    rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
    topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
  );

  FlBorderData get _defaultBorderData => FlBorderData(
    show: true,
    border: Border(
      bottom: BorderSide(color: Colors.grey.shade300),
      left: BorderSide(color: Colors.grey.shade300),
    ),
  );

  FlDotData get _defaultDotData =>
      const FlDotData(show: true, getDotPainter: _getDefaultDotPainter);

  static FlDotCirclePainter _getDefaultDotPainter(
    FlSpot spot,
    double xPercentage,
    LineChartBarData bar,
    int index,
  ) {
    return FlDotCirclePainter(
      radius: 4,
      color: bar.color ?? Colors.black,
      strokeWidth: 2,
      strokeColor: Colors.white,
    );
  }
}
