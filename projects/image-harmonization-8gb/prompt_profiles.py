"""Explicit opt-in prompt profiles. These instructions do not enforce spatial masks."""


def build_prompt(profile='legacy', subject=None, lighting=None, no_shadow=False, custom_prompt=None):
    if profile not in ('legacy','photometric'):
        raise ValueError('Unknown lighting profile')
    if profile == 'legacy':
        if subject is not None or lighting is not None:
            raise ValueError('--subject/--lighting require --lighting-profile photometric')
        prompt = ('Adjust the pasted foreground object lighting and color to match the surrounding scene. '
                  'Preserve every object shape, material, texture and background geometry exactly. ')
        prompt += ('Do not add any shadows.' if no_shadow else 'Add a subtle natural contact and cast shadow attached to the object on the ground, consistent with scene lighting.')
        return custom_prompt or prompt
    if custom_prompt is not None:
        raise ValueError('--prompt cannot be combined with --lighting-profile photometric')
    if subject is None or lighting is None or not subject.strip() or not lighting.strip():
        raise ValueError('photometric requires non-empty --subject and --lighting')
    prompt = (f'Relight only the {subject.strip()} to match {lighting.strip()}. '
              'Adjust the foreground subject exposure, white balance, color cast, and shading so that it looks photographed in this scene, not separately lit. '
              'Its current brightness and color cast are allowed to change; keep its identity, material colors, silhouette, pose and fine detail. '
              'Keep the background exposure, white balance, colors, contrast, scenery and camera framing unchanged. '
              'Do not apply a global color grade to the photograph. ')
    return prompt + ('Do not add or alter cast shadows in this test.' if no_shadow else
                     'Also add a subtle natural contact shadow attached to its feet or base, consistent with the existing light direction.')
