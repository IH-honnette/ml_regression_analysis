import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const SalaryApp());

/// Base URL of the deployed FastAPI service (Render).
const String kApiBaseUrl = 'https://salary-prediction-api-8tzr.onrender.com';

const double kInrToRwf = 15.28;

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


class FieldSpec {
  final String key;
  final String label;
  final String hint;
  final double min;
  final double max;
  final String help;
  final bool isInt;

  const FieldSpec(this.key, this.label, this.hint, this.min, this.max, this.help,
      {this.isInt = false});
}

class FormSection {
  final String title;
  final IconData icon;
  final List<FieldSpec> fields;
  const FormSection(this.title, this.icon, this.fields);
}

const List<FormSection> kSections = [
  FormSection('Education', Icons.school_outlined, [
    FieldSpec('CollegeTier', 'College Tier', '1 (top) or 2', 1, 2,
        'Tier of the college: 1 = top-ranked institution, 2 = other. A lower '
        'tier usually links to higher pay.',
        isInt: true),
    FieldSpec('collegeGPA', 'College GPA (%)', '0 – 100', 0, 100,
        'Final college GPA expressed as a percentage (0–100).'),
  ]),
  FormSection('Aptitude Scores', Icons.psychology_alt_outlined, [
    FieldSpec('English', 'English Score', '0 – 900', 0, 900,
        'AMCAT English aptitude test score (typically 180–900).'),
    FieldSpec('Logical', 'Logical Score', '0 – 900', 0, 900,
        'AMCAT Logical reasoning test score (typically 195–800).'),
    FieldSpec('Quant', 'Quant Score', '0 – 900', 0, 900,
        'AMCAT Quantitative ability score (typically 120–900). This is the '
        'strongest single predictor of salary.'),
    FieldSpec('Domain', 'Domain Score', '0.0 – 1.0', 0, 1,
        'Normalized domain-knowledge score for the student\'s specialization '
        '(0 = weak, 1 = strong).'),
    FieldSpec('ComputerProgramming', 'Computer Programming', '0 – 900', 0, 900,
        'AMCAT Computer Programming module score (0–900).'),
  ]),
  FormSection('Personality (Big Five)', Icons.emoji_people_outlined, [
    FieldSpec('conscientiousness', 'Conscientiousness', '-8 – 8', -8, 8,
        'How organized and dependable the person is (standardized score, '
        'roughly -8 to 8).'),
    FieldSpec('agreeableness', 'Agreeableness', '-8 – 8', -8, 8,
        'How cooperative and empathetic the person is (standardized score).'),
    FieldSpec('extraversion', 'Extraversion', '-8 – 8', -8, 8,
        'How outgoing and sociable the person is (standardized score).'),
    FieldSpec('nueroticism', 'Neuroticism', '-8 – 8', -8, 8,
        'Emotional sensitivity / tendency toward anxiety (standardized score).'),
    FieldSpec('openess_to_experience', 'Openness to Experience', '-8 – 8', -8, 8,
        'Curiosity and openness to new ideas (standardized score).'),
  ]),
];

final List<FieldSpec> kFields = [for (final s in kSections) ...s.fields];

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
  final Map<String, FocusNode> _focusNodes = {
    for (final f in kFields) f.key: FocusNode(),
  };

  bool _loading = false;
  String? _result;
  String? _resultRwf;
  bool _isError = false;
  String? _focusedKey;

  @override
  void initState() {
    super.initState();
    for (final f in kFields) {
      _focusNodes[f.key]!.addListener(() {
        if (_focusNodes[f.key]!.hasFocus) {
          setState(() => _focusedKey = f.key);
        } else if (_focusedKey == f.key) {
          setState(() => _focusedKey = null);
        }
      });
    }
  }

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    for (final n in _focusNodes.values) {
      n.dispose();
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
      _resultRwf = null;
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
        final rwf = data['predicted_salary_rwf'] != null
            ? (data['predicted_salary_rwf'] as num).toDouble()
            : salary * kInrToRwf;
        setState(() {
          _isError = false;
          _result = '₹ ${_thousands(salary)}';
          _resultRwf = 'RWF ${_thousands(rwf)}';
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
      _resultRwf = null;
      _isError = false;
    });
  }

  FieldSpec? get _focusedField {
    if (_focusedKey == null) return null;
    for (final f in kFields) {
      if (f.key == _focusedKey) return f;
    }
    return null;
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
            child: Column(
              children: [
                _buildInfoBanner(context),
                Expanded(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          ...kSections.map(_buildSection),
                          const SizedBox(height: 8),
                          _buildActions(),
                          const SizedBox(height: 20),
                          _buildResult(),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildInfoBanner(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final field = _focusedField;
    final active = field != null;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: active
            ? scheme.primaryContainer
            : scheme.surfaceContainerHighest.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
            color: active
                ? scheme.primary.withValues(alpha: 0.4)
                : Colors.transparent),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(active ? Icons.info_outline : Icons.touch_app_outlined,
              size: 20,
              color: active ? scheme.primary : scheme.onSurfaceVariant),
          const SizedBox(width: 10),
          Expanded(
            child: active
                ? Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(field.label,
                          style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: scheme.onPrimaryContainer)),
                      const SizedBox(height: 2),
                      Text(field.help,
                          style: TextStyle(
                              fontSize: 13, color: scheme.onPrimaryContainer)),
                    ],
                  )
                : Text(
                    'Tap any field to see what it means. Fill all fields, then '
                    'press Predict.',
                    style: TextStyle(
                        fontSize: 13, color: scheme.onSurfaceVariant)),
          ),
        ],
      ),
    );
  }

  Widget _buildSection(FormSection section) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.only(bottom: 10, top: 4),
            child: Row(
              children: [
                Icon(section.icon, size: 20, color: scheme.primary),
                const SizedBox(width: 8),
                Text(section.title,
                    style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: scheme.primary)),
                const SizedBox(width: 10),
                Expanded(child: Divider(color: scheme.outlineVariant)),
              ],
            ),
          ),
          ...section.fields.map(_buildField),
        ],
      ),
    );
  }

  Widget _buildField(FieldSpec f) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextFormField(
        controller: _controllers[f.key],
        focusNode: _focusNodes[f.key],
        onTap: () => setState(() => _focusedKey = f.key),
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

  Widget _buildActions() {
    return Row(
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
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
          ),
          child: const Text('Clear'),
        ),
      ],
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
          if (!_isError && _resultRwf != null) ...[
            const SizedBox(height: 12),
            Divider(color: border, height: 1),
            const SizedBox(height: 12),
            Text('≈ $_resultRwf',
                textAlign: TextAlign.center,
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w600,
                    color: textColor)),
            const Text('Rwandan Francs',
                style: TextStyle(fontSize: 13, color: Colors.black54)),
          ],
        ],
      ),
    );
  }
}
