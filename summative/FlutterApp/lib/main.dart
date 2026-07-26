import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const SalaryApp());

/// Base URL of the deployed FastAPI service.
/// After deploying to Render, replace this with your public URL, e.g.
/// https://salary-prediction-api.onrender.com
const String kApiBaseUrl = 'https://salary-prediction-api.onrender.com';

class SalaryApp extends StatelessWidget {
  const SalaryApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Salary Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF3457D5),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          isDense: true,
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

/// One input field definition: label, hint, and valid numeric range.
class FieldSpec {
  final String key;
  final String label;
  final String hint;
  final double min;
  final double max;
  final bool isInt;

  const FieldSpec(this.key, this.label, this.hint, this.min, this.max,
      {this.isInt = false});
}

const List<FieldSpec> kFields = [
  FieldSpec('CollegeTier', 'College Tier', '1 (top) or 2', 1, 2, isInt: true),
  FieldSpec('collegeGPA', 'College GPA (%)', '0 – 100', 0, 100),
  FieldSpec('English', 'English Score', '0 – 900', 0, 900),
  FieldSpec('Logical', 'Logical Score', '0 – 900', 0, 900),
  FieldSpec('Quant', 'Quant Score', '0 – 900', 0, 900),
  FieldSpec('Domain', 'Domain Score', '0.0 – 1.0', 0, 1),
  FieldSpec('ComputerProgramming', 'Computer Programming', '0 – 900', 0, 900),
  FieldSpec('conscientiousness', 'Conscientiousness', '-8 – 8', -8, 8),
  FieldSpec('agreeableness', 'Agreeableness', '-8 – 8', -8, 8),
  FieldSpec('extraversion', 'Extraversion', '-8 – 8', -8, 8),
  FieldSpec('nueroticism', 'Neuroticism', '-8 – 8', -8, 8),
  FieldSpec('openess_to_experience', 'Openness to Experience', '-8 – 8', -8, 8),
];

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  final Map<String, TextEditingController> _controllers = {
    for (final f in kFields) f.key: TextEditingController(),
  };

  bool _loading = false;
  String? _result;
  bool _isError = false;

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  String? _validate(FieldSpec f, String? value) {
    if (value == null || value.trim().isEmpty) {
      return 'Required';
    }
    final n = double.tryParse(value.trim());
    if (n == null) return 'Enter a number';
    if (n < f.min || n > f.max) {
      return 'Must be between ${_fmt(f.min)} and ${_fmt(f.max)}';
    }
    return null;
  }

  String _fmt(double v) => v == v.roundToDouble() ? v.toInt().toString() : '$v';

  Future<void> _predict() async {
    setState(() {
      _result = null;
      _isError = false;
    });
    if (!_formKey.currentState!.validate()) {
      setState(() {
        _isError = true;
        _result = 'Please fix the highlighted fields before predicting.';
      });
      return;
    }

    final body = <String, dynamic>{};
    for (final f in kFields) {
      final n = double.parse(_controllers[f.key]!.text.trim());
      body[f.key] = f.isInt ? n.toInt() : n;
    }

    setState(() => _loading = true);
    try {
      final resp = await http
          .post(
            Uri.parse('$kApiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 60));

      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        final salary = (data['predicted_salary'] as num).toDouble();
        setState(() {
          _isError = false;
          _result = '₹ ${_thousands(salary)}';
        });
      } else if (resp.statusCode == 422) {
        final data = jsonDecode(resp.body);
        final detail = data['detail'];
        final msg = detail is List && detail.isNotEmpty
            ? '${detail[0]['loc'].last}: ${detail[0]['msg']}'
            : 'Validation error';
        setState(() {
          _isError = true;
          _result = 'Invalid input — $msg';
        });
      } else {
        setState(() {
          _isError = true;
          _result = 'Server error (${resp.statusCode}). Try again.';
        });
      }
    } catch (e) {
      setState(() {
        _isError = true;
        _result = 'Could not reach the API. Check your connection.\n$e';
      });
    } finally {
      setState(() => _loading = false);
    }
  }

  String _thousands(double v) {
    final s = v.round().toString();
    final buf = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      if (i > 0 && (s.length - i) % 3 == 0) buf.write(',');
      buf.write(s[i]);
    }
    return buf.toString();
  }

  void _clear() {
    for (final c in _controllers.values) {
      c.clear();
    }
    setState(() {
      _result = null;
      _isError = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Graduate Salary Predictor'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 560),
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Text(
                      'Enter a graduate’s academic, aptitude and '
                      'personality scores to estimate their starting salary.',
                      style: TextStyle(fontSize: 14, color: Colors.black54),
                    ),
                    const SizedBox(height: 16),
                    ...kFields.map(_buildField),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: FilledButton.icon(
                            onPressed: _loading ? null : _predict,
                            icon: _loading
                                ? const SizedBox(
                                    width: 18,
                                    height: 18,
                                    child: CircularProgressIndicator(
                                        strokeWidth: 2, color: Colors.white),
                                  )
                                : const Icon(Icons.calculate),
                            label: Text(_loading ? 'Predicting...' : 'Predict'),
                            style: FilledButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 16),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        OutlinedButton(
                          onPressed: _loading ? null : _clear,
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(
                                vertical: 16, horizontal: 16),
                          ),
                          child: const Text('Clear'),
                        ),
                      ],
                    ),
                    const SizedBox(height: 20),
                    _buildResult(),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildField(FieldSpec f) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextFormField(
        controller: _controllers[f.key],
        keyboardType: TextInputType.numberWithOptions(
            decimal: !f.isInt, signed: f.min < 0),
        inputFormatters: [
          FilteringTextInputFormatter.allow(RegExp(r'[0-9.\-]')),
        ],
        decoration: InputDecoration(
          labelText: f.label,
          hintText: f.hint,
          helperText: 'Range: ${_fmt(f.min)} to ${_fmt(f.max)}',
        ),
        validator: (v) => _validate(f, v),
      ),
    );
  }

  Widget _buildResult() {
    if (_result == null) return const SizedBox.shrink();
    final color = _isError ? Colors.red.shade50 : Colors.green.shade50;
    final border = _isError ? Colors.red.shade200 : Colors.green.shade300;
    final textColor = _isError ? Colors.red.shade900 : Colors.green.shade900;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color,
        border: Border.all(color: border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Text(
            _isError ? 'Error' : 'Predicted Starting Salary',
            style: TextStyle(
                fontSize: 14, fontWeight: FontWeight.w600, color: textColor),
          ),
          const SizedBox(height: 8),
          Text(
            _result!,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: _isError ? 15 : 30,
              fontWeight: FontWeight.bold,
              color: textColor,
            ),
          ),
          if (!_isError)
            const Padding(
              padding: EdgeInsets.only(top: 4),
              child: Text('per year (INR)',
                  style: TextStyle(fontSize: 13, color: Colors.black54)),
            ),
        ],
      ),
    );
  }
}
