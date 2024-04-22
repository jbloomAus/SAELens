# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update



# Checklist:

- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have not rewritten tests relating to key interfaces which would affect backward compatibility

<!--
As you go through the checklist above, you can mark something as done by putting an x character in it

For example,
- [x] I have done this task
- [ ] I have not done this task
-->

### You have tested formatting, typing and unit tests (acceptance tests not currently in use)

- [ ] I have run `make check-ci` to check format and linting. (you can run `make format` to format code if needed.)

### Performance Check.

If you have implemented a training change, please indicate precisely how performance changes with respect to the following metrics:
- [ ] L0
- [ ] CE Loss
- [ ] MSE Loss
- [ ] Feature Dashboard Interpretability

Please links to wandb dashboards with a control and test group. 