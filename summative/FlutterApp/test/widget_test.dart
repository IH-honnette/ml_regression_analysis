import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:salary_predictor/main.dart';

void main() {
  testWidgets('renders 12 input fields and a Predict button',
      (WidgetTester tester) async {
    await tester.pumpWidget(const SalaryApp());

    // One TextFormField per model feature.
    expect(find.byType(TextFormField), findsNWidgets(kFields.length));

    // The Predict button is present.
    expect(find.widgetWithText(FilledButton, 'Predict'), findsOneWidget);

    // The result area starts empty (shown only after a prediction).
    expect(find.text('Predicted Starting Salary'), findsNothing);
  });
}
